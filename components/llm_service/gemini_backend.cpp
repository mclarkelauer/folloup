// Gemini backend: Google Generative Language API. Text runs through generateContent /
// countTokens; transcription is a resumable file upload followed by a generateContent call that
// references the uploaded file.
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

constexpr const char* kTag = "GeminiBackend";
constexpr const char* kLabel = "Gemini";
constexpr const char* kApiBaseUrl = "https://generativelanguage.googleapis.com/v1beta/";
constexpr const char* kUploadUrl =
    "https://generativelanguage.googleapis.com/upload/v1beta/files";
constexpr const char* kAudioMimeType = "audio/wav";
constexpr const char* kTranscriptPrompt =
    "Generate a verbatim transcript of the speech in this audio. Respond with transcript text "
    "only. Do not add commentary or formatting.";
constexpr int kAuthTimeoutMs = 15000;
constexpr int kGenerateTimeoutMs = 60000;  // text generation can be slow for large prompts
constexpr int kTranscribeTimeoutMs = 30000;
constexpr size_t kHttpUploadChunkSamples = 2048;

std::string DefaultModel()
{
#if defined(CONFIG_FOLLOWUP_GEMINI_MODEL)
    const std::string configured = http::TrimCopy(CONFIG_FOLLOWUP_GEMINI_MODEL);
    if (!configured.empty()) {
        return configured;
    }
#endif
    return "models/gemini-2.5-flash-lite";
}

std::vector<http::Header> JsonHeaders(const std::string& api_key)
{
    return {
        {"x-goog-api-key", api_key},
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };
}

// Gemini error envelope: {"error":{"status":"RESOURCE_EXHAUSTED","message":"..."}}.
void PopulateHttpError(cJSON* root, const http::Response& response, std::string* error_code,
                       std::string* error_message)
{
    *error_code = "http_error";
    error_message->clear();
    if (root != nullptr) {
        cJSON* error = cJSON_GetObjectItemCaseSensitive(root, "error");
        if (cJSON_IsObject(error)) {
            const std::string status = http::JsonStringField(error, "status");
            const std::string message = http::JsonStringField(error, "message");
            if (!status.empty()) {
                *error_code = status;
            }
            if (!message.empty()) {
                *error_message = message;
            }
        }
    }
    if (error_message->empty()) {
        *error_message = response.body.empty() ? "Gemini request failed" : response.body;
    }
    *error_message = http::TrimForLog(std::move(*error_message));
}

// A single text-part prompt: {"contents":[{"parts":[{"text": prompt}]}]}. temperature=0 keeps
// summaries deterministic; countTokens ignores generationConfig, so it is harmless there.
std::string BuildTextRequestBody(const std::string& prompt, bool include_generation_config)
{
    cJSON* root = cJSON_CreateObject();
    cJSON* contents = cJSON_AddArrayToObject(root, "contents");
    cJSON* content = cJSON_CreateObject();
    cJSON_AddItemToArray(contents, content);
    cJSON* parts = cJSON_AddArrayToObject(content, "parts");
    cJSON* prompt_part = cJSON_CreateObject();
    cJSON_AddStringToObject(prompt_part, "text", prompt.c_str());
    cJSON_AddItemToArray(parts, prompt_part);
    if (include_generation_config) {
        cJSON* generation_config = cJSON_AddObjectToObject(root, "generationConfig");
        cJSON_AddNumberToObject(generation_config, "temperature", 0);
    }
    const std::string body = http::JsonToString(root);
    cJSON_Delete(root);
    return body;
}

// Concatenate the text of every part in candidates[0].content.parts[].
std::string ExtractCandidateText(cJSON* root)
{
    if (root == nullptr) {
        return {};
    }
    cJSON* candidates = cJSON_GetObjectItemCaseSensitive(root, "candidates");
    if (!cJSON_IsArray(candidates)) {
        return {};
    }
    cJSON* candidate = cJSON_GetArrayItem(candidates, 0);
    if (!cJSON_IsObject(candidate)) {
        return {};
    }
    cJSON* content = cJSON_GetObjectItemCaseSensitive(candidate, "content");
    if (!cJSON_IsObject(content)) {
        return {};
    }
    cJSON* parts = cJSON_GetObjectItemCaseSensitive(content, "parts");
    if (!cJSON_IsArray(parts)) {
        return {};
    }
    std::string text;
    cJSON* part = nullptr;
    cJSON_ArrayForEach(part, parts)
    {
        text += http::JsonStringField(part, "text");
    }
    return text;
}

std::string BuildTranscriptRequestJson(const std::string& file_uri)
{
    cJSON* root = cJSON_CreateObject();
    cJSON* contents = cJSON_AddArrayToObject(root, "contents");
    cJSON* content = cJSON_CreateObject();
    cJSON_AddItemToArray(contents, content);
    cJSON* parts = cJSON_AddArrayToObject(content, "parts");

    cJSON* prompt_part = cJSON_CreateObject();
    cJSON_AddStringToObject(prompt_part, "text", kTranscriptPrompt);
    cJSON_AddItemToArray(parts, prompt_part);

    cJSON* audio_part = cJSON_CreateObject();
    cJSON* file_data = cJSON_AddObjectToObject(audio_part, "fileData");
    cJSON_AddStringToObject(file_data, "mimeType", kAudioMimeType);
    cJSON_AddStringToObject(file_data, "fileUri", file_uri.c_str());
    cJSON_AddItemToArray(parts, audio_part);

    cJSON* generation_config = cJSON_AddObjectToObject(root, "generationConfig");
    cJSON_AddNumberToObject(generation_config, "temperature", 0);

    const std::string json = http::JsonToString(root);
    cJSON_Delete(root);
    return json;
}

class GeminiBackendImpl : public Backend {
 public:
    Provider provider() const override { return Provider::kGemini; }
    std::string TextModel() const override { return DefaultModel(); }
    std::string TranscriptionModel() const override { return DefaultModel(); }

    AuthResult Authenticate(const std::string& api_key) override
    {
        AuthResult result = {};
        const std::string url = std::string(kApiBaseUrl) + TextModel();
        const http::Response http = http::Perform(
            url, HTTP_METHOD_GET,
            {{"x-goog-api-key", api_key}, {"Accept", "application/json"}}, {}, kAuthTimeoutMs,
            kLabel);
        result.http_status = http.status_code;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.success = true;
            result.model_resource_name = http::JsonStringField(root, "name");
            result.model_display_name = http::JsonStringField(root, "displayName");
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
        const std::string url = std::string(kApiBaseUrl) + TextModel() + ":generateContent";
        const http::Response http =
            http::Perform(url, HTTP_METHOD_POST, JsonHeaders(api_key),
                          BuildTextRequestBody(prompt, true), kGenerateTimeoutMs, kLabel);
        result.http_status = http.status_code;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.text = ExtractCandidateText(root);
            result.success = !result.text.empty();
            if (!result.success) {
                result.error_code = "empty_response";
                result.error_message = "Gemini returned no text";
            }
        } else {
            PopulateHttpError(root, http, &result.error_code, &result.error_message);
        }
        if (root != nullptr) {
            cJSON_Delete(root);
        }
        return result;
    }

    TokenCountResult CountTokens(const std::string& api_key, const std::string& prompt) override
    {
        TokenCountResult result = {};
        const std::string url = std::string(kApiBaseUrl) + TextModel() + ":countTokens";
        const http::Response http =
            http::Perform(url, HTTP_METHOD_POST, JsonHeaders(api_key),
                          BuildTextRequestBody(prompt, false), kGenerateTimeoutMs, kLabel);
        result.http_status = http.status_code;
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            cJSON* total =
                root != nullptr ? cJSON_GetObjectItemCaseSensitive(root, "totalTokens") : nullptr;
            if (cJSON_IsNumber(total)) {
                result.total_tokens = total->valueint;
                result.success = true;
            } else {
                result.error_code = "empty_response";
                result.error_message = "Gemini returned no token count";
            }
        } else {
            PopulateHttpError(root, http, &result.error_code, &result.error_message);
        }
        if (root != nullptr) {
            cJSON_Delete(root);
        }
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

        const int64_t task_started_us = esp_timer_get_time();

        // 1. Start a resumable upload to obtain an upload URL.
        const std::string content_length = std::to_string(clip.wav_byte_count());
        const http::Response upload_start = http::Perform(
            kUploadUrl, HTTP_METHOD_POST,
            {
                {"x-goog-api-key", api_key},
                {"X-Goog-Upload-Protocol", "resumable"},
                {"X-Goog-Upload-Command", "start"},
                {"X-Goog-Upload-Header-Content-Length", content_length},
                {"X-Goog-Upload-Header-Content-Type", kAudioMimeType},
                {"Content-Type", "application/json"},
            },
            "{\"file\":{\"display_name\":\"FOLLOWUP_NOTE\"}}", kTranscribeTimeoutMs, kLabel);
        if (!upload_start.error_code.empty() || upload_start.upload_url.empty() ||
            upload_start.status_code < 200 || upload_start.status_code >= 300) {
            result.http_status = upload_start.status_code;
            result.error_code = upload_start.error_code.empty() ? "upload_start_failed"
                                                                : upload_start.error_code;
            result.error_message = http::TrimForLog(
                !upload_start.error_message.empty()
                    ? upload_start.error_message
                    : (!upload_start.body.empty() ? upload_start.body
                                                  : "Failed to start Gemini file upload"));
            return result;
        }

        // 2. Stream the WAV (header + PCM chunks) and finalize the upload.
        const int64_t upload_started_us = esp_timer_get_time();
        const http::Response upload_finalize = http::PerformWavUpload(
            upload_start.upload_url,
            {
                {"Accept", "application/json"},
                {"X-Goog-Upload-Offset", "0"},
                {"X-Goog-Upload-Command", "upload, finalize"},
            },
            {}, clip, {}, kTranscribeTimeoutMs, kLabel);
        result.http_status = upload_finalize.status_code;
        result.upload_elapsed_ms =
            static_cast<uint64_t>((esp_timer_get_time() - upload_started_us) / 1000ULL);
        if (!upload_finalize.error_code.empty() || upload_finalize.status_code < 200 ||
            upload_finalize.status_code >= 300) {
            result.error_code = upload_finalize.error_code.empty() ? "upload_finalize_failed"
                                                                   : upload_finalize.error_code;
            result.error_message = http::TrimForLog(
                !upload_finalize.error_message.empty()
                    ? upload_finalize.error_message
                    : (!upload_finalize.body.empty() ? upload_finalize.body
                                                     : "Failed to upload Gemini audio file"));
            return result;
        }

        cJSON* file_root =
            cJSON_ParseWithLength(upload_finalize.body.c_str(), upload_finalize.body.size());
        const std::string file_uri = http::JsonNestedStringField(file_root, "file", "uri");
        if (file_root != nullptr) {
            cJSON_Delete(file_root);
        }
        if (file_uri.empty()) {
            result.error_code = "file_uri_missing";
            result.error_message = "Gemini upload did not return a file URI";
            return result;
        }

        // 3. generateContent referencing the uploaded file.
        const std::string url = std::string(kApiBaseUrl) + TranscriptionModel() + ":generateContent";
        const http::Response http =
            http::Perform(url, HTTP_METHOD_POST, JsonHeaders(api_key),
                          BuildTranscriptRequestJson(file_uri), kGenerateTimeoutMs, kLabel);
        result.http_status = http.status_code;
        result.total_elapsed_ms =
            static_cast<uint64_t>((esp_timer_get_time() - task_started_us) / 1000ULL);
        if (!http.error_code.empty()) {
            result.error_code = http.error_code;
            result.error_message = http::TrimForLog(http.error_message);
            return result;
        }

        cJSON* root = cJSON_ParseWithLength(http.body.c_str(), http.body.size());
        if (http.status_code >= 200 && http.status_code < 300) {
            result.transcript = http::TrimForLog(ExtractCandidateText(root), 1U << 20);
            result.success = !result.transcript.empty();
            if (!result.success) {
                result.error_code = "empty_transcript";
                result.error_message = "Gemini returned no transcript text";
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

Backend& GeminiBackend()
{
    static GeminiBackendImpl instance;
    return instance;
}

}  // namespace backend
}  // namespace llm_service
