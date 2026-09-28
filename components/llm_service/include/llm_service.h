#ifndef LLM_SERVICE_H_
#define LLM_SERVICE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "esp_err.h"
#include "esp_http_server.h"

namespace recording_service {
class RecordedClip;
}

// Provider-agnostic AI service. One active provider (Muse via the Meta Model API, or Gemini)
// handles speech-to-text, text generation, and optional token counting. Every provider keeps its
// own API key, so switching providers never discards a stored key.
namespace llm_service {

enum class Provider : uint8_t {
    kGemini = 0,
    kMuse = 1,
};
constexpr size_t kProviderCount = 2;

enum class ApiKeySource : uint8_t {
    kNone = 0,
    kSdkConfig,
    kNvs,
};

// Key + model state for one provider, whether or not it is the active one.
struct ProviderKeyState {
    Provider provider = Provider::kGemini;
    bool has_key = false;  // a stored or built-in key is present
    bool has_stored_api_key = false;
    bool has_sdkconfig_api_key = false;
    ApiKeySource api_key_source = ApiKeySource::kNone;
    std::string api_key_last4;
    std::string model_name;                // text / summary model
    std::string transcription_model_name;  // speech-to-text model
};

struct SettingsSnapshot {
    Provider provider = Provider::kMuse;  // active provider
    bool configured = false;              // the active provider has a key
    bool has_stored_api_key = false;
    bool has_sdkconfig_api_key = false;
    ApiKeySource api_key_source = ApiKeySource::kNone;
    std::string api_key_last4;
    std::string model_name;
    std::string transcription_model_name;
    std::array<ProviderKeyState, kProviderCount> providers = {};
};

struct RuntimeSnapshot {
    bool initialized = false;
    bool ready = false;
    bool request_in_flight = false;
    bool auth_checked = false;
    bool authenticated = false;
    bool supports_audio_understanding = false;
    bool supports_structured_output = false;
    int last_http_status = 0;
    std::string last_status_message;
    std::string last_model_resource_name;
    std::string last_model_display_name;
    std::string last_error_code;
    std::string last_error_message;
};

struct Snapshot {
    SettingsSnapshot settings = {};
    RuntimeSnapshot runtime = {};
};

struct Event {
    Snapshot snapshot = {};
};

struct SettingsPatch {
    bool has_provider = false;  // switch the active provider
    Provider provider = Provider::kMuse;
    bool has_api_key = false;  // store a key
    std::string api_key;
    bool has_api_key_provider = false;  // which provider the key belongs to (default: active)
    Provider api_key_provider = Provider::kMuse;
};

struct Result {
    bool success = false;
    bool validation_error = false;
    int status_code = 500;
    std::string field;
    std::string error_code;
    std::string message;
};

// Result of a synchronous text-generation call.
struct TextResult {
    bool success = false;
    int http_status = 0;
    std::string text = {};
    std::string error_code = {};
    std::string error_message = {};
};

// Result of a synchronous token-count call. Providers without a token-count endpoint return
// success=false with error_code "unsupported"; callers fall back to a local estimate.
struct TokenCountResult {
    bool success = false;
    int http_status = 0;
    int total_tokens = 0;
    std::string error_code = {};
    std::string error_message = {};
};

// Result of a synchronous audio transcription call.
struct TranscriptionResult {
    bool success = false;
    int http_status = 0;
    std::string transcript = {};
    std::string error_code = {};
    std::string error_message = {};
    uint32_t clip_duration_ms = 0;
    size_t wav_bytes = 0;
    uint32_t upload_chunk_count = 0;
    uint64_t upload_elapsed_ms = 0;
    uint64_t total_elapsed_ms = 0;
};

using EventHandler = void (*)(const Event& event, void* context);

esp_err_t Init();
void SetEventHandler(EventHandler handler, void* context);
Snapshot GetSnapshot();

Result ApplySettingsPatch(const SettingsPatch& patch);
Result ClearStoredApiKey();  // active provider
Result ClearStoredApiKeyFor(Provider provider);

bool HasApiKey();
Provider GetActiveProvider();
std::string GetEffectiveApiKey();
std::string GetEffectiveModelName();
std::string GetProviderDisplayName();  // active provider, e.g. "Muse"

// Synchronous provider calls (block on HTTP; run them from a worker task, never a UI/input
// task). They use the active provider's key + model and return the parsed result or an error.
TextResult GenerateText(const std::string& prompt);
TokenCountResult CountTokens(const std::string& prompt);
TranscriptionResult Transcribe(const recording_service::RecordedClip& clip);
bool BeginAuthentication();
void SetNetworkState(bool connected, bool access_point_mode);
void RegisterPortalRoutes(httpd_handle_t server);

const char* ApiKeySourceName(ApiKeySource source);
const char* ProviderId(Provider provider);           // "gemini" / "muse"
const char* ProviderDisplayName(Provider provider);  // "Gemini" / "Muse"
bool ParseProviderId(const std::string& id, Provider* out);
// True for the rate-limit / quota error codes any provider can return.
bool IsQuotaErrorCode(const std::string& error_code);

}  // namespace llm_service

#endif  // LLM_SERVICE_H_
