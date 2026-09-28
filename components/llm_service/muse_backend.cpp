// Muse backend: Meta Model API. Text runs through the OpenAI-compatible chat-completions
// endpoint with a Muse Spark model; transcription is a single multipart POST of the WAV clip to
// the Muse Voice Transcribe endpoint. The API has no token-count endpoint, so CountTokens
// reports "unsupported" and callers fall back to a local estimate.
#include <string>
#include <vector>

#include "cJSON.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "llm_backend.h"
#include "llm_http.h"
#include "recording_service.h"
#include "sdkconfig.h"

namespace llm_service {
namespace backend {
namespace {

constexpr const char* kTag = "MuseBackend";
constexpr const char* kLabel = "Muse";
constexpr const char* kApiBaseUrl = "https://api.meta.ai/v1/";
constexpr const char* kTranscriptionModel = "muse-voice-transcribe-1.0";
// Meta's documented file-transcription request. DIARIZATION returns the full transcript plus
// per-speaker turns; a single-speaker memo simply yields one speaker.
constexpr const char* kTranscriptionMode = "DIARIZATION";
constexpr const char* kMultipartBoundary = "----FolloupMuseBoundary7f3a9c1e";
constexpr int kMaxCompletionTokens = 4096;
constexpr int kAuthTimeoutMs = 15000;
constexpr int kGenerateTimeoutMs = 60000;
constexpr int kTranscribeTimeoutMs = 45000;
constexpr size_t kHttpUploadChunkSamples = 2048;

std::string DefaultTextModel()
{
#if defined(CONFIG_FOLLOWUP_MUSE_TEXT_MODEL)
    const std::string configured = http::TrimCopy(CONFIG_FOLLOWUP_MUSE_TEXT_MODEL);
    if (!configured.empty()) {
        return configured;
    }
#endif
    return "muse-spark-1.3";
}

std::string BearerHeader(const std::string& api_key)
{
    return "Bearer " + api_key;
}

std::vector<http::Header> JsonHeaders(const std::string& api_key)
{
    return {
        {"Authorization", BearerHeader(api_key)},
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };
}

// OpenAI-style envelope: {"error":{"code":"invalid_api_key","type":"authentication_error",
// "message":"Unauthorized"}}. A 429 without a code is reported as rate_limit_exceeded so the
// UI can show a quota toast.
void PopulateHttpError(cJSON* root, const http::Response& response, std::string* error_code,
                       std::string* error_message)
{
    *error_code = "http_error";
    error_message->clear();
    if (root != nullptr) {
        cJSON* error = cJSON_GetObjectItemCaseSensitive(root, "error");
        if (cJSON_IsObject(error)) {
            const std::string code = http::JsonStringField(error, "code");
            const std::string type = http::JsonStringField(error, "type");
            const std::string message = http::JsonStringField(error, "message");
            if (!code.empty()) {
                *error_code = code;
            } else if (!type.empty()) {
                *error_code = type;
            }
            if (!message.empty()) {
                *error_message = message;
            }
        }
    }
    if (*error_code == "http_error" && response.status_code == 429) {
        *error_code = "rate_limit_exceeded";
    }
    if (error_message->empty()) {
        *error_message = response.body.empty() ? "Muse request failed" : response.body;
    }
    *error_message = http::TrimForLog(std::move(*error_message));
}

// {"model":..., "messages":[{"role":"user","content": prompt}], "max_completion_tokens": N}
std::string BuildChatRequestBody(const std::string& model, const std::string& prompt)
{
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", model.c_str());
    cJSON* messages = cJSON_AddArrayToObject(root, "messages");
    cJSON* message = cJSON_CreateObject();
    cJSON_AddStringToObject(message, "role", "user");
    cJSON_AddStringToObject(message, "content", prompt.c_str());
    cJSON_AddItemToArray(messages, message);
    cJSON_AddNumberToObject(root, "max_completion_tokens", kMaxCompletionTokens);
    const std::string body = http::JsonToString(root);
    cJSON_Delete(root);
    return body;
}

// choices[0].message.content, which is a string or (for some models) an array of text parts.
std::string ExtractChatText(cJSON* root)
{
    if (root == nullptr) {
        return {};
    }
    cJSON* choices = cJSON_GetObjectItemCaseSensitive(root, "choices");
    if (!cJSON_IsArray(choices)) {
        return {};
    }
    cJSON* choice = cJSON_GetArrayItem(choices, 0);
    if (!cJSON_IsObject(choice)) {
        return {};
    }
    cJSON* message = cJSON_GetObjectItemCaseSensitive(choice, "message");
    if (!cJSON_IsObject(message)) {
        return {};
    }
    cJSON* content = cJSON_GetObjectItemCaseSensitive(message, "content");
    if (cJSON_IsString(content) && content->valuestring != nullptr) {
        return content->valuestring;
    }
    std::string text;
    if (cJSON_IsArray(content)) {
        cJSON* part = nullptr;
        cJSON_ArrayForEach(part, content)
        {
            text += http::JsonStringField(part, "text");
        }
    }
    return text;
}

std::string BuildTranscriptionRequestJson()
{
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "mode", kTranscriptionMode);
    cJSON_AddStringToObject(root, "model", kTranscriptionModel);
    cJSON_AddStringToObject(root, "audioEncoding", "WAV");
    const std::string json = http::JsonToString(root);
    cJSON_Delete(root);
    return json;
}

// The response carries the whole transcript in "transcript"; fall back to joining the
// per-speaker "turns" if that field is ever empty.
std::string ExtractTranscript(cJSON* root)
{
    if (root == nullptr) {
        return {};
    }
    std::string transcript = http::TrimCopy(http::JsonStringField(root, "transcript"));
    if (!transcript.empty()) {
        return transcript;
    }
    cJSON* turns = cJSON_GetObjectItemCaseSensitive(root, "turns");
    if (!cJSON_IsArray(turns)) {
        return {};
    }
    cJSON* turn = nullptr;
    cJSON_ArrayForEach(turn, turns)
    {
        const std::string piece = http::TrimCopy(http::JsonStringField(turn, "transcript"));
        if (piece.empty()) {
            continue;
        }
        if (!transcript.empty()) {
            transcript += ' ';
        }
        transcript += piece;
    }
    return transcript;
}

class MuseBackendImpl : public Backend {
 public:
    Provider provider() const override { return Provider::kMuse; }
    std::string TextModel() const override { return DefaultTextModel(); }
    std::string TranscriptionModel() const override { return kTranscriptionModel; }

    AuthResult Authenticate(const std::string& api_key) override
    {
        AuthResult result = {};
        const std::string model = TextModel();
        const std::string url = std::string(kApiBaseUrl) + "models/" + model;
        const http::Response http = http::Perform(
            url, HTTP_METHOD_GET,
            {{"Authorization", BearerHeader(api_key)}, {"Accept", "application/json"}}, {},
            kAuthTimeoutMs, kLabel);
        result.http_status = http.status_code;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.success = true;
            const std::string id = http::JsonStringField(root, "id");
            result.model_resource_name = id.empty() ? model : id;
            result.model_display_name = result.model_resource_name;
        } else {
            PopulateHttpError(root, http, &result.error_code, &result.error_message);
        }
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return result;
    }

    TextResult GenerateText(const std::string& api_key, const std::string& prompt) override
    {
        TextResult result = {};
        const std::string url = std::string(kApiBaseUrl) + "chat/completions";
        const http::Response http =
            http::Perform(url, HTTP_METHOD_POST, JsonHeaders(api_key),
                          BuildChatRequestBody(TextModel(), prompt), kGenerateTimeoutMs, kLabel);
        result.http_status = http.status_code;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.text = ExtractChatText(root);
            result.success = !result.text.empty();
            if (!result.success) {
                result.error_code = "empty_response";
                result.error_message = "Muse returned no text";
            }
        } else {
            PopulateHttpError(root, http, &result.error_code, &result.error_message);
        }
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return result;
    }

    TokenCountResult CountTokens(const std::string& /*api_key*/,
                                 const std::string& /*prompt*/) override
    {
        TokenCountResult result = {};
        result.error_code = "unsupported";
        result.error_message = "Muse has no token-count endpoint";
        return result;
    }

    TranscriptionResult Transcribe(const std::string& api_key,
                                   const recording_service::RecordedClip& clip) override
    {
        TranscriptionResult result = {};
        result.clip_duration_ms = clip.duration_ms();
        result.wav_bytes = clip.wav_byte_count();
        result.upload_chunk_count = static_cast<uint32_t>(
            (clip.sample_count() + kHttpUploadChunkSamples - 1U) / kHttpUploadChunkSamples);

        // Multipart envelope: a JSON "request" part, then the WAV as the "audio" file part.
        std::string prefix;
        prefix += "--";
        prefix += kMultipartBoundary;
        prefix += "\r\nContent-Disposition: form-data; name=\"request\"\r\n";
        prefix += "Content-Type: application/json\r\n\r\n";
        prefix += BuildTranscriptionRequestJson();
        prefix += "\r\n--";
        prefix += kMultipartBoundary;
        prefix += "\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"clip.wav\"\r\n";
        prefix += "Content-Type: audio/wav\r\n\r\n";
        std::string suffix = "\r\n--";
        suffix += kMultipartBoundary;
        suffix += "--\r\n";

        const std::string url = std::string(kApiBaseUrl) + "asr/transcribe";
        const int64_t started_us = esp_timer_get_time();
        const http::Response http = http::PerformWavUpload(
            url,
            {
                {"Authorization", BearerHeader(api_key)},
                {"Content-Type", std::string("multipart/form-data; boundary=") + kMultipartBoundary},
                {"Accept", "application/json"},
            },
            prefix, clip, suffix, kTranscribeTimeoutMs, kLabel);
        result.http_status = http.status_code;
        result.upload_elapsed_ms =
            static_cast<uint64_t>((esp_timer_get_time() - started_us) / 1000ULL);
        result.total_elapsed_ms = result.upload_elapsed_ms;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.transcript = http::TrimForLog(ExtractTranscript(root), 1U << 20);
            result.success = !result.transcript.empty();
            if (!result.success) {
                result.error_code = "empty_transcript";
                result.error_message = "Muse returned no transcript text";
            }
        } else {
            PopulateHttpError(root, http, &result.error_code, &result.error_message);
        }
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return result;
    }
};

}  // namespace

Backend& MuseBackend()
{
    static MuseBackendImpl instance;
    return instance;
}

Backend& BackendFor(Provider provider)
{
    switch (provider) {
        case Provider::kMuse:
            return MuseBackend();
        case Provider::kGemini:
        default:
            return GeminiBackend();
    }
}

}  // namespace backend
}  // namespace llm_service
