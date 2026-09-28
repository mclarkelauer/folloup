#include "llm_service.h"

#include <array>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>

#include "cJSON.h"
#include "esp_log.h"
#include "followup_task_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "llm_backend.h"
#include "llm_http.h"
#include "nvs.h"
#include "recording_service.h"
#include "sdkconfig.h"

namespace llm_service {
namespace {

constexpr const char* kTag = "LlmService";
constexpr const char* kSettingsTag = "LlmSettings";
constexpr const char* kStorageNamespace = "llm";
constexpr const char* kStorageProviderKey = "provider";
// The pre-provider firmware kept the Gemini key under gemini/api_key; it is imported once.
constexpr const char* kLegacyGeminiNamespace = "gemini";
constexpr const char* kLegacyGeminiApiKey = "api_key";
constexpr const char* kPortalApiSettingsUri = "/api/settings/llm";
constexpr const char* kPortalApiSettingsResetUri = "/api/settings/llm/reset";
constexpr const char* kPortalApiRuntimeUri = "/api/runtime/llm";
constexpr size_t kMaxPortalPayloadLen = 768;
constexpr uint32_t kAuthTaskStackWords = 8192;

constexpr std::array<Provider, kProviderCount> kAllProviders = {Provider::kGemini,
                                                                 Provider::kMuse};

struct AuthTaskContext {
    Provider provider = Provider::kMuse;
    std::string api_key;
    uint32_t generation = 0;
};

std::mutex s_mutex;
EventHandler s_event_handler = nullptr;
void* s_event_context = nullptr;
bool s_initialized = false;
bool s_network_connected = false;
bool s_access_point_mode = false;
bool s_request_in_flight = false;
bool s_auth_checked = false;
bool s_authenticated = false;
uint32_t s_auth_generation = 0;
int s_last_http_status = 0;
Provider s_active_provider = Provider::kMuse;
std::array<std::string, kProviderCount> s_stored_api_keys = {};
std::string s_last_status_message;
std::string s_last_model_resource_name;
std::string s_last_model_display_name;
std::string s_last_error_code;
std::string s_last_error_message;

size_t ProviderIndex(Provider provider)
{
    const size_t index = static_cast<size_t>(provider);
    return index < kProviderCount ? index : 0;
}

const char* StorageKeyFor(Provider provider)
{
    switch (provider) {
        case Provider::kMuse:
            return "key_muse";
        case Provider::kGemini:
        default:
            return "key_gemini";
    }
}

std::string SdkConfigApiKey(Provider provider)
{
    switch (provider) {
        case Provider::kMuse:
#if defined(CONFIG_FOLLOWUP_MUSE_API_KEY)
            return http::TrimCopy(CONFIG_FOLLOWUP_MUSE_API_KEY);
#else
            return {};
#endif
        case Provider::kGemini:
        default:
#if defined(CONFIG_FOLLOWUP_GEMINI_API_KEY)
            return http::TrimCopy(CONFIG_FOLLOWUP_GEMINI_API_KEY);
#else
            return {};
#endif
    }
}

Provider DefaultProvider()
{
#if defined(CONFIG_FOLLOWUP_LLM_PROVIDER)
    Provider provider = Provider::kMuse;
    if (ParseProviderId(http::TrimCopy(CONFIG_FOLLOWUP_LLM_PROVIDER), &provider)) {
        return provider;
    }
#endif
    return Provider::kMuse;
}

std::string ReadNvsString(nvs_handle_t handle, const char* key)
{
    size_t size = 0;
    if (nvs_get_str(handle, key, nullptr, &size) != ESP_OK || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (nvs_get_str(handle, key, value.data(), &size) != ESP_OK) {
        return {};
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::string LoadLegacyGeminiKey()
{
    nvs_handle_t handle = 0;
    if (nvs_open(kLegacyGeminiNamespace, NVS_READONLY, &handle) != ESP_OK) {
        return {};
    }
    const std::string key = http::TrimCopy(ReadNvsString(handle, kLegacyGeminiApiKey));
    nvs_close(handle);
    return key;
}

bool WriteNvsString(const char* key, const std::string& value)
{
    nvs_handle_t handle = 0;
    esp_err_t err = nvs_open(kStorageNamespace, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(kSettingsTag, "Failed to open NVS namespace for write: %s", esp_err_to_name(err));
        return false;
    }
    err = nvs_set_str(handle, key, value.c_str());
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }
    nvs_close(handle);
    if (err != ESP_OK) {
        ESP_LOGE(kSettingsTag, "Failed to save %s: %s", key, esp_err_to_name(err));
        return false;
    }
    return true;
}

bool EraseNvsKey(const char* key)
{
    nvs_handle_t handle = 0;
    esp_err_t err = nvs_open(kStorageNamespace, NVS_READWRITE, &handle);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        return true;
    }
    if (err != ESP_OK) {
        ESP_LOGE(kSettingsTag, "Failed to open NVS namespace for clear: %s", esp_err_to_name(err));
        return false;
    }
    err = nvs_erase_key(handle, key);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        err = ESP_OK;
    }
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }
    nvs_close(handle);
    if (err != ESP_OK) {
        ESP_LOGE(kSettingsTag, "Failed to clear %s: %s", key, esp_err_to_name(err));
        return false;
    }
    return true;
}

// Loads the active provider and every stored key. Must be called with s_mutex held.
void LoadSettingsLocked()
{
    s_active_provider = DefaultProvider();
    for (std::string& key : s_stored_api_keys) {
        key.clear();
    }

    nvs_handle_t handle = 0;
    const esp_err_t err = nvs_open(kStorageNamespace, NVS_READONLY, &handle);
    if (err == ESP_OK) {
        Provider stored_provider = Provider::kMuse;
        if (ParseProviderId(ReadNvsString(handle, kStorageProviderKey), &stored_provider)) {
            s_active_provider = stored_provider;
        }
        for (Provider provider : kAllProviders) {
            s_stored_api_keys[ProviderIndex(provider)] =
                http::TrimCopy(ReadNvsString(handle, StorageKeyFor(provider)));
        }
        nvs_close(handle);
    } else if (err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(kSettingsTag, "Failed to open NVS namespace: %s", esp_err_to_name(err));
    }

    // One-time import of a key saved by the Gemini-only firmware.
    std::string& gemini_key = s_stored_api_keys[ProviderIndex(Provider::kGemini)];
    if (gemini_key.empty()) {
        const std::string legacy = LoadLegacyGeminiKey();
        if (!legacy.empty() && WriteNvsString(StorageKeyFor(Provider::kGemini), legacy)) {
            gemini_key = legacy;
            ESP_LOGI(kSettingsTag, "Imported the Gemini API key from the legacy namespace");
        }
    }
}

void EnsureLoadedLocked()
{
    if (!s_initialized) {
        LoadSettingsLocked();
        s_initialized = true;
    }
}

std::string EffectiveApiKeyLocked(Provider provider)
{
    const std::string& stored = s_stored_api_keys[ProviderIndex(provider)];
    if (!stored.empty()) {
        return stored;
    }
    return SdkConfigApiKey(provider);
}

std::string ActiveApiKeyLocked()
{
    return EffectiveApiKeyLocked(s_active_provider);
}

ApiKeySource ApiKeySourceLocked(Provider provider)
{
    if (!s_stored_api_keys[ProviderIndex(provider)].empty()) {
        return ApiKeySource::kNvs;
    }
    if (!SdkConfigApiKey(provider).empty()) {
        return ApiKeySource::kSdkConfig;
    }
    return ApiKeySource::kNone;
}

std::string ApiKeyLast4Locked(Provider provider)
{
    const std::string key = EffectiveApiKeyLocked(provider);
    if (key.size() < 4) {
        return {};
    }
    return key.substr(key.size() - 4);
}

void SetLastErrorLocked(const char* error_code, const char* message)
{
    s_last_error_code = error_code != nullptr ? error_code : "";
    s_last_error_message = message != nullptr ? message : "";
}

void ClearLastErrorLocked()
{
    s_last_error_code.clear();
    s_last_error_message.clear();
}

// Drops any in-flight / completed authentication so the next network or settings change
// re-probes the active provider's key.
void ResetAuthStateLocked(const std::string& status_message)
{
    ++s_auth_generation;
    s_request_in_flight = false;
    s_auth_checked = false;
    s_authenticated = false;
    s_last_http_status = 0;
    s_last_status_message = status_message;
    s_last_model_resource_name.clear();
    s_last_model_display_name.clear();
    ClearLastErrorLocked();
}

ProviderKeyState BuildProviderKeyStateLocked(Provider provider)
{
    backend::Backend& impl = backend::BackendFor(provider);
    ProviderKeyState state = {};
    state.provider = provider;
    state.has_stored_api_key = !s_stored_api_keys[ProviderIndex(provider)].empty();
    state.has_sdkconfig_api_key = !SdkConfigApiKey(provider).empty();
    state.has_key = state.has_stored_api_key || state.has_sdkconfig_api_key;
    state.api_key_source = ApiKeySourceLocked(provider);
    state.api_key_last4 = ApiKeyLast4Locked(provider);
    state.model_name = impl.TextModel();
    state.transcription_model_name = impl.TranscriptionModel();
    return state;
}

Snapshot BuildSnapshotLocked()
{
    Snapshot snapshot = {};
    for (Provider provider : kAllProviders) {
        snapshot.settings.providers[ProviderIndex(provider)] =
            BuildProviderKeyStateLocked(provider);
    }
    const ProviderKeyState& active =
        snapshot.settings.providers[ProviderIndex(s_active_provider)];
    snapshot.settings.provider = s_active_provider;
    snapshot.settings.configured = active.has_key;
    snapshot.settings.has_stored_api_key = active.has_stored_api_key;
    snapshot.settings.has_sdkconfig_api_key = active.has_sdkconfig_api_key;
    snapshot.settings.api_key_source = active.api_key_source;
    snapshot.settings.api_key_last4 = active.api_key_last4;
    snapshot.settings.model_name = active.model_name;
    snapshot.settings.transcription_model_name = active.transcription_model_name;

    snapshot.runtime.initialized = s_initialized;
    snapshot.runtime.ready = active.has_key && s_authenticated;
    snapshot.runtime.request_in_flight = s_request_in_flight;
    snapshot.runtime.auth_checked = s_auth_checked;
    snapshot.runtime.authenticated = s_authenticated;
    snapshot.runtime.supports_audio_understanding = false;
    snapshot.runtime.supports_structured_output = false;
    snapshot.runtime.last_http_status = s_last_http_status;
    snapshot.runtime.last_status_message = s_last_status_message;
    snapshot.runtime.last_model_resource_name = s_last_model_resource_name;
    snapshot.runtime.last_model_display_name = s_last_model_display_name;
    snapshot.runtime.last_error_code = s_last_error_code;
    snapshot.runtime.last_error_message = s_last_error_message;
    return snapshot;
}

void Notify()
{
    EventHandler handler = nullptr;
    void* context = nullptr;
    Event event = {};
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        handler = s_event_handler;
        context = s_event_context;
        event.snapshot = BuildSnapshotLocked();
    }
    if (handler != nullptr) {
        handler(event, context);
    }
}

bool ShouldStartAuthenticationLocked()
{
    return s_initialized && s_network_connected && !s_request_in_flight && !s_authenticated &&
           !ActiveApiKeyLocked().empty();
}

void CompleteAuthentication(uint32_t generation, Provider provider,
                            const backend::AuthResult& result)
{
    bool stale_result = false;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        stale_result = generation != s_auth_generation;
        if (!stale_result) {
            s_request_in_flight = false;
            s_auth_checked = true;
            s_last_http_status = result.http_status;
            if (!result.success) {
                s_authenticated = false;
                s_last_status_message = "Authentication failed";
                s_last_model_resource_name.clear();
                s_last_model_display_name.clear();
                SetLastErrorLocked(result.error_code.c_str(), result.error_message.c_str());
            } else {
                s_authenticated = true;
                s_last_model_resource_name = result.model_resource_name;
                s_last_model_display_name = result.model_display_name;
                s_last_status_message =
                    std::string("Authenticated with ") +
                    (!s_last_model_display_name.empty() ? s_last_model_display_name
                                                        : ProviderDisplayName(provider));
                ClearLastErrorLocked();
            }
        }
    }

    if (stale_result) {
        ESP_LOGI(kTag, "Ignoring stale %s authentication result for generation %lu",
                 ProviderDisplayName(provider), static_cast<unsigned long>(generation));
        return;
    }

    if (!result.success) {
        ESP_LOGW(kTag, "%s authentication failed: http=%d code=%s message=%s",
                 ProviderDisplayName(provider), result.http_status,
                 result.error_code.empty() ? "http_error" : result.error_code.c_str(),
                 result.error_message.empty() ? "unknown" : result.error_message.c_str());
    } else {
        ESP_LOGI(kTag, "%s authentication succeeded: model=%s http=%d",
                 ProviderDisplayName(provider),
                 result.model_resource_name.empty() ? "unknown"
                                                    : result.model_resource_name.c_str(),
                 result.http_status);
    }
    Notify();
}

void AuthenticationTask(void* arg)
{
    std::unique_ptr<AuthTaskContext> context(static_cast<AuthTaskContext*>(arg));
    if (context == nullptr) {
        backend::AuthResult result = {};
        result.error_code = "task_context_missing";
        result.error_message = "Authentication task context missing";
        CompleteAuthentication(0, Provider::kMuse, result);
        vTaskDelete(nullptr);
        return;
    }
    const backend::AuthResult result =
        backend::BackendFor(context->provider).Authenticate(context->api_key);
    CompleteAuthentication(context->generation, context->provider, result);
    vTaskDelete(nullptr);
}

void MaybeBeginAuthentication()
{
    bool should_start = false;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        should_start = ShouldStartAuthenticationLocked();
    }
    if (should_start) {
        (void)BeginAuthentication();
    }
}

Result MakeResult(bool success, bool validation_error, int status_code, const char* field,
                  const char* error_code, const std::string& message)
{
    Result result = {};
    result.success = success;
    result.validation_error = validation_error;
    result.status_code = status_code;
    result.field = field != nullptr ? field : "";
    result.error_code = error_code != nullptr ? error_code : "";
    result.message = message;
    return result;
}

// ---- Portal (HTTP) plumbing --------------------------------------------------------------

std::string ReadRequestBody(httpd_req_t* request)
{
    if (request == nullptr || request->content_len <= 0) {
        return {};
    }
    std::string body(static_cast<size_t>(request->content_len), '\0');
    size_t offset = 0;
    while (offset < body.size()) {
        const int received = httpd_req_recv(request, body.data() + offset, body.size() - offset);
        if (received <= 0) {
            return {};
        }
        offset += static_cast<size_t>(received);
    }
    return body;
}

esp_err_t SendJsonResponse(httpd_req_t* request, int status_code, cJSON* root)
{
    if (request == nullptr) {
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return ESP_FAIL;
    }
    const std::string payload = http::JsonToString(root);
    if (root != nullptr) {
        cJSON_Delete(root);
    }
    switch (status_code) {
        case 200:
            httpd_resp_set_status(request, HTTPD_200);
            break;
        case 400:
            httpd_resp_set_status(request, HTTPD_400);
            break;
        case 500:
        default:
            httpd_resp_set_status(request, HTTPD_500);
            break;
    }
    httpd_resp_set_type(request, "application/json; charset=utf-8");
    return httpd_resp_send(request, payload.c_str(), payload.size());
}

esp_err_t SendErrorResponse(httpd_req_t* request, const Result& result)
{
    cJSON* root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "success", false);
    cJSON_AddStringToObject(root, "message", result.message.c_str());
    cJSON_AddStringToObject(root, "error_code", result.error_code.c_str());
    cJSON_AddStringToObject(root, "field", result.field.c_str());
    return SendJsonResponse(request, result.status_code, root);
}

void AppendProviderKeyState(cJSON* parent, const ProviderKeyState& state)
{
    cJSON* node = cJSON_AddObjectToObject(parent, ProviderId(state.provider));
    cJSON_AddStringToObject(node, "display_name", ProviderDisplayName(state.provider));
    cJSON_AddBoolToObject(node, "has_key", state.has_key);
    cJSON_AddStringToObject(node, "last4", state.api_key_last4.c_str());
    cJSON_AddBoolToObject(node, "has_stored_api_key", state.has_stored_api_key);
    cJSON_AddBoolToObject(node, "has_sdkconfig_api_key", state.has_sdkconfig_api_key);
    cJSON_AddStringToObject(node, "api_key_source", ApiKeySourceName(state.api_key_source));
    cJSON_AddStringToObject(node, "model_name", state.model_name.c_str());
    cJSON_AddStringToObject(node, "transcription_model_name",
                            state.transcription_model_name.c_str());
}

void AppendSnapshot(cJSON* root, const Snapshot& snapshot, const char* message)
{
    cJSON_AddBoolToObject(root, "success", true);
    cJSON_AddStringToObject(root, "message", message != nullptr ? message : "");

    cJSON* settings = cJSON_AddObjectToObject(root, "settings");
    cJSON_AddStringToObject(settings, "provider", ProviderId(snapshot.settings.provider));
    cJSON_AddStringToObject(settings, "provider_display_name",
                            ProviderDisplayName(snapshot.settings.provider));
    cJSON_AddBoolToObject(settings, "configured", snapshot.settings.configured);
    cJSON_AddBoolToObject(settings, "has_key", snapshot.settings.configured);
    cJSON_AddStringToObject(settings, "last4", snapshot.settings.api_key_last4.c_str());
    cJSON_AddBoolToObject(settings, "has_stored_api_key", snapshot.settings.has_stored_api_key);
    cJSON_AddBoolToObject(settings, "has_sdkconfig_api_key",
                          snapshot.settings.has_sdkconfig_api_key);
    cJSON_AddStringToObject(settings, "api_key_source",
                            ApiKeySourceName(snapshot.settings.api_key_source));
    cJSON_AddStringToObject(settings, "api_key_last4", snapshot.settings.api_key_last4.c_str());
    cJSON_AddStringToObject(settings, "model_name", snapshot.settings.model_name.c_str());
    cJSON_AddStringToObject(settings, "transcription_model_name",
                            snapshot.settings.transcription_model_name.c_str());
    cJSON* providers = cJSON_AddObjectToObject(settings, "providers");
    for (const ProviderKeyState& state : snapshot.settings.providers) {
        AppendProviderKeyState(providers, state);
    }

    cJSON* runtime = cJSON_AddObjectToObject(root, "runtime");
    cJSON_AddBoolToObject(runtime, "initialized", snapshot.runtime.initialized);
    cJSON_AddBoolToObject(runtime, "ready", snapshot.runtime.ready);
    cJSON_AddBoolToObject(runtime, "request_in_flight", snapshot.runtime.request_in_flight);
    cJSON_AddBoolToObject(runtime, "auth_checked", snapshot.runtime.auth_checked);
    cJSON_AddBoolToObject(runtime, "authenticated", snapshot.runtime.authenticated);
    cJSON_AddBoolToObject(runtime, "supports_audio_understanding",
                          snapshot.runtime.supports_audio_understanding);
    cJSON_AddBoolToObject(runtime, "supports_structured_output",
                          snapshot.runtime.supports_structured_output);
    cJSON_AddNumberToObject(runtime, "last_http_status", snapshot.runtime.last_http_status);
    cJSON_AddStringToObject(runtime, "last_status_message",
                            snapshot.runtime.last_status_message.c_str());
    cJSON_AddStringToObject(runtime, "last_model_resource_name",
                            snapshot.runtime.last_model_resource_name.c_str());
    cJSON_AddStringToObject(runtime, "last_model_display_name",
                            snapshot.runtime.last_model_display_name.c_str());
    cJSON_AddStringToObject(runtime, "last_error_code", snapshot.runtime.last_error_code.c_str());
    cJSON_AddStringToObject(runtime, "last_error_message",
                            snapshot.runtime.last_error_message.c_str());
}

// Reads an optional provider id field; returns false (with `error` set) when it is present but
// invalid.
bool ParseOptionalProvider(cJSON* root, const char* key, bool* present, Provider* out,
                           std::string* error)
{
    *present = false;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (item == nullptr || cJSON_IsNull(item)) {
        return true;
    }
    if (!cJSON_IsString(item) || item->valuestring == nullptr ||
        !ParseProviderId(http::TrimCopy(item->valuestring), out)) {
        *error = std::string("Invalid ") + key + " (expected \"muse\" or \"gemini\")";
        return false;
    }
    *present = true;
    return true;
}

bool ParsePatchBody(const std::string& body, SettingsPatch* patch, std::string* error)
{
    cJSON* root = cJSON_ParseWithLength(body.c_str(), body.size());
    if (root == nullptr) {
        *error = "Invalid JSON body";
        return false;
    }
    bool ok = ParseOptionalProvider(root, "provider", &patch->has_provider, &patch->provider,
                                    error) &&
              ParseOptionalProvider(root, "key_provider", &patch->has_api_key_provider,
                                    &patch->api_key_provider, error);
    if (ok) {
        cJSON* api_key = cJSON_GetObjectItemCaseSensitive(root, "api_key");
        if (cJSON_IsString(api_key) && api_key->valuestring != nullptr) {
            patch->has_api_key = true;
            patch->api_key = api_key->valuestring;
        } else if (api_key != nullptr && !cJSON_IsNull(api_key)) {
            *error = "Invalid api_key";
            ok = false;
        }
    }
    cJSON_Delete(root);
    return ok;
}

esp_err_t RegisterPortalRoute(httpd_handle_t server, const httpd_uri_t* handler)
{
    if (server == nullptr || handler == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    const esp_err_t err = httpd_register_uri_handler(server, handler);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "Failed to register portal route %s [%d]: %s",
                 handler->uri != nullptr ? handler->uri : "<null>",
                 static_cast<int>(handler->method), esp_err_to_name(err));
    }
    return err;
}

esp_err_t HandlePortalSettingsGet(httpd_req_t* request)
{
    cJSON* root = cJSON_CreateObject();
    AppendSnapshot(root, GetSnapshot(), "AI settings loaded");
    return SendJsonResponse(request, 200, root);
}

esp_err_t HandlePortalSettingsPatch(httpd_req_t* request)
{
    if (request == nullptr || request->content_len <= 0 ||
        request->content_len > static_cast<int>(kMaxPortalPayloadLen)) {
        return SendErrorResponse(request, MakeResult(false, true, 400, "", "invalid_payload",
                                                     "Invalid AI settings payload"));
    }

    const std::string body = ReadRequestBody(request);
    SettingsPatch patch = {};
    std::string parse_error;
    if (body.empty() || !ParsePatchBody(body, &patch, &parse_error)) {
        return SendErrorResponse(
            request, MakeResult(false, true, 400, "", "invalid_payload",
                                parse_error.empty() ? "Invalid AI settings payload" : parse_error));
    }
    if (!patch.has_provider && !patch.has_api_key) {
        return SendErrorResponse(request, MakeResult(false, true, 400, "", "empty_patch",
                                                     "Nothing to update"));
    }

    const Result result = ApplySettingsPatch(patch);
    if (!result.success) {
        return SendErrorResponse(request, result);
    }
    cJSON* root = cJSON_CreateObject();
    AppendSnapshot(root, GetSnapshot(), result.message.c_str());
    return SendJsonResponse(request, 200, root);
}

esp_err_t HandlePortalSettingsReset(httpd_req_t* request)
{
    Provider provider = GetActiveProvider();
    if (request != nullptr && request->content_len > 0 &&
        request->content_len <= static_cast<int>(kMaxPortalPayloadLen)) {
        const std::string body = ReadRequestBody(request);
        cJSON* root = cJSON_ParseWithLength(body.c_str(), body.size());
        if (root != nullptr) {
            bool present = false;
            std::string error;
            const bool ok = ParseOptionalProvider(root, "provider", &present, &provider, &error);
            cJSON_Delete(root);
            if (!ok) {
                return SendErrorResponse(
                    request, MakeResult(false, true, 400, "provider", "invalid_provider", error));
            }
        }
    }

    const Result result = ClearStoredApiKeyFor(provider);
    if (!result.success) {
        return SendErrorResponse(request, result);
    }
    cJSON* root = cJSON_CreateObject();
    AppendSnapshot(root, GetSnapshot(), result.message.c_str());
    return SendJsonResponse(request, 200, root);
}

esp_err_t HandlePortalRuntimeGet(httpd_req_t* request)
{
    cJSON* root = cJSON_CreateObject();
    AppendSnapshot(root, GetSnapshot(), "AI runtime loaded");
    return SendJsonResponse(request, 200, root);
}

}  // namespace

esp_err_t Init()
{
    Snapshot snapshot = {};
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_initialized) {
            return ESP_OK;
        }
        LoadSettingsLocked();
        s_last_status_message = ActiveApiKeyLocked().empty()
                                    ? std::string("No ") + ProviderDisplayName(s_active_provider) +
                                          " API key configured"
                                    : std::string(ProviderDisplayName(s_active_provider)) +
                                          " API key available";
        ClearLastErrorLocked();
        s_initialized = true;
        snapshot = BuildSnapshotLocked();
    }

    ESP_LOGI(kTag, "AI service initialized: provider=%s configured=%d source=%s key_last4=%s",
             ProviderId(snapshot.settings.provider), snapshot.settings.configured ? 1 : 0,
             ApiKeySourceName(snapshot.settings.api_key_source),
             snapshot.settings.api_key_last4.empty() ? "none"
                                                     : snapshot.settings.api_key_last4.c_str());
    Notify();
    MaybeBeginAuthentication();
    return ESP_OK;
}

void SetEventHandler(EventHandler handler, void* context)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    s_event_handler = handler;
    s_event_context = context;
}

Snapshot GetSnapshot()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    return BuildSnapshotLocked();
}

Result ApplySettingsPatch(const SettingsPatch& patch)
{
    bool should_start_auth = false;
    Result failure = {};
    bool failed = false;
    std::string message;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        EnsureLoadedLocked();

        if (!patch.has_provider && !patch.has_api_key) {
            return MakeResult(false, true, 400, "", "empty_patch", "Nothing to update");
        }

        if (patch.has_api_key) {
            const Provider target =
                patch.has_api_key_provider ? patch.api_key_provider : s_active_provider;
            const std::string display = ProviderDisplayName(target);
            const std::string trimmed = http::TrimCopy(patch.api_key);
            if (trimmed.empty()) {
                return MakeResult(false, true, 400, "api_key", "invalid_api_key",
                                  display + " API key is required");
            }
            if (!WriteNvsString(StorageKeyFor(target), trimmed)) {
                SetLastErrorLocked("nvs_write_failed", "Failed to store API key");
                failure = MakeResult(false, false, 500, "api_key", "nvs_write_failed",
                                     "Failed to store " + display + " API key");
                failed = true;
            } else {
                s_stored_api_keys[ProviderIndex(target)] = trimmed;
                message = display + " API key stored";
            }
        }

        if (!failed && patch.has_provider && patch.provider != s_active_provider) {
            if (!WriteNvsString(kStorageProviderKey, ProviderId(patch.provider))) {
                SetLastErrorLocked("nvs_write_failed", "Failed to store provider");
                failure = MakeResult(false, false, 500, "provider", "nvs_write_failed",
                                     "Failed to store the provider choice");
                failed = true;
            } else {
                s_active_provider = patch.provider;
                if (!message.empty()) {
                    message += "; ";
                }
                message += std::string("Switched to ") + ProviderDisplayName(patch.provider);
            }
        }

        if (!failed) {
            if (message.empty()) {
                message = "Settings unchanged";
            }
            ResetAuthStateLocked(message);
            should_start_auth = ShouldStartAuthenticationLocked();
        }
    }

    Notify();
    if (failed) {
        return failure;
    }
    if (should_start_auth) {
        (void)BeginAuthentication();
    }
    return MakeResult(true, false, 200, nullptr, nullptr, message);
}

Result ClearStoredApiKey()
{
    return ClearStoredApiKeyFor(GetActiveProvider());
}

Result ClearStoredApiKeyFor(Provider provider)
{
    bool should_start_auth = false;
    bool clear_failed = false;
    const std::string display = ProviderDisplayName(provider);
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        EnsureLoadedLocked();
        if (!EraseNvsKey(StorageKeyFor(provider))) {
            SetLastErrorLocked("nvs_clear_failed", "Failed to clear API key");
            clear_failed = true;
        } else {
            s_stored_api_keys[ProviderIndex(provider)].clear();
            if (provider == s_active_provider) {
                ResetAuthStateLocked(display + " API key cleared");
                should_start_auth = ShouldStartAuthenticationLocked();
            }
        }
    }

    Notify();
    if (clear_failed) {
        return MakeResult(false, false, 500, "api_key", "nvs_clear_failed",
                          "Failed to clear " + display + " API key");
    }
    if (should_start_auth) {
        (void)BeginAuthentication();
    }
    return MakeResult(true, false, 200, nullptr, nullptr, display + " API key cleared");
}

bool HasApiKey()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    EnsureLoadedLocked();
    return !ActiveApiKeyLocked().empty();
}

Provider GetActiveProvider()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    EnsureLoadedLocked();
    return s_active_provider;
}

std::string GetEffectiveApiKey()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    EnsureLoadedLocked();
    return ActiveApiKeyLocked();
}

std::string GetEffectiveModelName()
{
    return backend::BackendFor(GetActiveProvider()).TextModel();
}

std::string GetProviderDisplayName()
{
    return ProviderDisplayName(GetActiveProvider());
}

namespace {

// Copies the active provider + key under the lock so a long HTTP call never holds it.
bool ResolveActive(Provider* provider, std::string* api_key)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    EnsureLoadedLocked();
    *provider = s_active_provider;
    *api_key = ActiveApiKeyLocked();
    return !api_key->empty();
}

}  // namespace

TextResult GenerateText(const std::string& prompt)
{
    Provider provider = Provider::kMuse;
    std::string api_key;
    if (!ResolveActive(&provider, &api_key)) {
        TextResult result = {};
        result.error_code = "not_configured";
        result.error_message = std::string("No ") + ProviderDisplayName(provider) +
                               " API key configured";
        return result;
    }
    if (prompt.empty()) {
        TextResult result = {};
        result.error_code = "empty_prompt";
        result.error_message = "Prompt was empty";
        return result;
    }

    const TextResult result = backend::BackendFor(provider).GenerateText(api_key, prompt);
    if (result.success) {
        ESP_LOGI(kTag, "%s text generation succeeded: http=%d chars=%u",
                 ProviderDisplayName(provider), result.http_status,
                 static_cast<unsigned>(result.text.size()));
    } else {
        ESP_LOGW(kTag, "%s text generation failed: http=%d code=%s message=%s",
                 ProviderDisplayName(provider), result.http_status,
                 result.error_code.empty() ? "<none>" : result.error_code.c_str(),
                 result.error_message.empty() ? "<none>" : result.error_message.c_str());
    }
    return result;
}

TokenCountResult CountTokens(const std::string& prompt)
{
    Provider provider = Provider::kMuse;
    std::string api_key;
    if (!ResolveActive(&provider, &api_key)) {
        TokenCountResult result = {};
        result.error_code = "not_configured";
        result.error_message = std::string("No ") + ProviderDisplayName(provider) +
                               " API key configured";
        return result;
    }
    if (prompt.empty()) {
        TokenCountResult result = {};
        result.success = true;  // an empty prompt is trivially zero tokens
        return result;
    }

    const TokenCountResult result = backend::BackendFor(provider).CountTokens(api_key, prompt);
    if (!result.success) {
        // Callers fall back to a size estimate, so this is debug-level to avoid chunking noise.
        ESP_LOGD(kTag, "%s token count unavailable: http=%d code=%s",
                 ProviderDisplayName(provider), result.http_status,
                 result.error_code.empty() ? "<none>" : result.error_code.c_str());
    }
    return result;
}

TranscriptionResult Transcribe(const recording_service::RecordedClip& clip)
{
    Provider provider = Provider::kMuse;
    std::string api_key;
    if (!ResolveActive(&provider, &api_key)) {
        TranscriptionResult result = {};
        result.clip_duration_ms = clip.duration_ms();
        result.wav_bytes = clip.wav_byte_count();
        result.error_code = "not_configured";
        result.error_message = std::string("No ") + ProviderDisplayName(provider) +
                               " API key configured";
        return result;
    }
    if (clip.empty()) {
        TranscriptionResult result = {};
        result.error_code = "empty_audio";
        result.error_message = "No recorded audio available";
        return result;
    }
    return backend::BackendFor(provider).Transcribe(api_key, clip);
}

bool BeginAuthentication()
{
    Provider provider = Provider::kMuse;
    std::string api_key;
    std::string api_key_last4;
    ApiKeySource api_key_source = ApiKeySource::kNone;
    uint32_t auth_generation = 0;
    bool missing_api_key = false;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        EnsureLoadedLocked();
        if (s_request_in_flight) {
            return false;
        }

        provider = s_active_provider;
        api_key = ActiveApiKeyLocked();
        if (api_key.empty()) {
            s_request_in_flight = false;
            s_auth_checked = false;
            s_authenticated = false;
            s_last_http_status = 0;
            s_last_status_message = "Authentication skipped";
            s_last_model_resource_name.clear();
            s_last_model_display_name.clear();
            const std::string message =
                std::string("No ") + ProviderDisplayName(provider) + " API key configured";
            SetLastErrorLocked("not_configured", message.c_str());
            missing_api_key = true;
        } else if (!s_network_connected) {
            return false;
        }

        if (!missing_api_key) {
            s_request_in_flight = true;
            s_auth_checked = false;
            s_authenticated = false;
            s_last_http_status = 0;
            s_last_status_message =
                std::string("Authenticating with ") + ProviderDisplayName(provider);
            s_last_model_resource_name.clear();
            s_last_model_display_name.clear();
            ClearLastErrorLocked();
            api_key_source = ApiKeySourceLocked(provider);
            api_key_last4 = ApiKeyLast4Locked(provider);
            auth_generation = ++s_auth_generation;
        }
    }

    Notify();
    if (missing_api_key) {
        return false;
    }

    std::unique_ptr<AuthTaskContext> context(new (std::nothrow) AuthTaskContext{});
    bool task_failed = context == nullptr;
    const char* failure_code = "task_alloc_failed";
    if (!task_failed) {
        context->provider = provider;
        context->api_key = std::move(api_key);
        context->generation = auth_generation;
        TaskHandle_t task_handle = nullptr;
        const BaseType_t created = xTaskCreatePinnedToCore(
            AuthenticationTask, "llm_auth", kAuthTaskStackWords, context.get(),
            followup_task_config::kPriorityGemini, &task_handle,
            followup_task_config::kSystemCore);
        if (created != pdPASS || task_handle == nullptr) {
            task_failed = true;
            failure_code = "task_start_failed";
        } else {
            context.release();
        }
    }

    if (task_failed) {
        {
            std::lock_guard<std::mutex> lock(s_mutex);
            s_request_in_flight = false;
            s_last_status_message = "Failed to start authentication";
            SetLastErrorLocked(failure_code, "Failed to start the authentication task");
        }
        Notify();
        return false;
    }

    ESP_LOGI(kTag, "Starting %s authentication (model=%s, source=%s, key_last4=%s)",
             ProviderDisplayName(provider), backend::BackendFor(provider).TextModel().c_str(),
             ApiKeySourceName(api_key_source),
             api_key_last4.empty() ? "none" : api_key_last4.c_str());
    return true;
}

void SetNetworkState(bool connected, bool access_point_mode)
{
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_network_connected = connected;
        s_access_point_mode = access_point_mode;
    }
    MaybeBeginAuthentication();
}

void RegisterPortalRoutes(httpd_handle_t server)
{
    if (server == nullptr) {
        return;
    }
    httpd_uri_t settings_get = {};
    settings_get.uri = kPortalApiSettingsUri;
    settings_get.method = HTTP_GET;
    settings_get.handler = HandlePortalSettingsGet;
    httpd_uri_t settings_patch = {};
    settings_patch.uri = kPortalApiSettingsUri;
    settings_patch.method = HTTP_PATCH;
    settings_patch.handler = HandlePortalSettingsPatch;
    httpd_uri_t settings_reset = {};
    settings_reset.uri = kPortalApiSettingsResetUri;
    settings_reset.method = HTTP_POST;
    settings_reset.handler = HandlePortalSettingsReset;
    httpd_uri_t runtime_get = {};
    runtime_get.uri = kPortalApiRuntimeUri;
    runtime_get.method = HTTP_GET;
    runtime_get.handler = HandlePortalRuntimeGet;

    if (RegisterPortalRoute(server, &settings_get) != ESP_OK ||
        RegisterPortalRoute(server, &settings_patch) != ESP_OK ||
        RegisterPortalRoute(server, &settings_reset) != ESP_OK ||
        RegisterPortalRoute(server, &runtime_get) != ESP_OK) {
        ESP_LOGW(kTag, "AI portal routes are incomplete");
    }
}

const char* ApiKeySourceName(ApiKeySource source)
{
    switch (source) {
        case ApiKeySource::kSdkConfig:
            return "sdkconfig";
        case ApiKeySource::kNvs:
            return "nvs";
        case ApiKeySource::kNone:
        default:
            return "none";
    }
}

const char* ProviderId(Provider provider)
{
    switch (provider) {
        case Provider::kMuse:
            return "muse";
        case Provider::kGemini:
        default:
            return "gemini";
    }
}

const char* ProviderDisplayName(Provider provider)
{
    switch (provider) {
        case Provider::kMuse:
            return "Muse";
        case Provider::kGemini:
        default:
            return "Gemini";
    }
}

bool ParseProviderId(const std::string& id, Provider* out)
{
    if (out == nullptr) {
        return false;
    }
    if (id == "muse") {
        *out = Provider::kMuse;
        return true;
    }
    if (id == "gemini") {
        *out = Provider::kGemini;
        return true;
    }
    return false;
}

bool IsQuotaErrorCode(const std::string& error_code)
{
    return error_code == "RESOURCE_EXHAUSTED" ||   // Gemini
           error_code == "rate_limit_exceeded" ||  // Muse / OpenAI-style
           error_code == "insufficient_quota";
}

}  // namespace llm_service
