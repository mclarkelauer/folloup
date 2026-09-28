#include "llm_http.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <strings.h>

#include "esp_crt_bundle.h"
#include "esp_log.h"
#include "recording_service.h"

namespace llm_service {
namespace http {
namespace {

constexpr const char* kUserAgent = "folloup";

esp_err_t EventHandler(esp_http_client_event_t* event)
{
    if (event == nullptr) {
        return ESP_FAIL;
    }
    auto* response = static_cast<Response*>(event->user_data);
    if (response == nullptr) {
        return ESP_OK;
    }
    if (event->event_id == HTTP_EVENT_ON_DATA && event->data != nullptr && event->data_len > 0) {
        response->body.append(static_cast<const char*>(event->data),
                              static_cast<size_t>(event->data_len));
    } else if (event->event_id == HTTP_EVENT_ON_HEADER && event->header_key != nullptr &&
               event->header_value != nullptr &&
               strcasecmp(event->header_key, "x-goog-upload-url") == 0) {
        response->upload_url = event->header_value;
    }
    return ESP_OK;
}

void ApplyHeaders(esp_http_client_handle_t client, const std::vector<Header>& headers)
{
    esp_http_client_set_header(client, "User-Agent", kUserAgent);
    for (const Header& header : headers) {
        esp_http_client_set_header(client, header.first.c_str(), header.second.c_str());
    }
}

bool ReadBody(esp_http_client_handle_t client, Response* response)
{
    std::array<char, 512> buffer = {};
    while (true) {
        const int read = esp_http_client_read(client, buffer.data(), buffer.size());
        if (read < 0) {
            response->error_code = "transport_error";
            response->error_message = "Failed reading HTTP response body";
            return false;
        }
        if (read == 0) {
            break;
        }
        response->body.append(buffer.data(), static_cast<size_t>(read));
    }
    return true;
}

void AppendLe16(uint16_t value, std::array<uint8_t, 44>* out, size_t* offset)
{
    if (out == nullptr || offset == nullptr || *offset + 2U > out->size()) {
        return;
    }
    (*out)[(*offset)++] = static_cast<uint8_t>(value & 0xFF);
    (*out)[(*offset)++] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

void AppendLe32(uint32_t value, std::array<uint8_t, 44>* out, size_t* offset)
{
    if (out == nullptr || offset == nullptr || *offset + 4U > out->size()) {
        return;
    }
    (*out)[(*offset)++] = static_cast<uint8_t>(value & 0xFF);
    (*out)[(*offset)++] = static_cast<uint8_t>((value >> 8) & 0xFF);
    (*out)[(*offset)++] = static_cast<uint8_t>((value >> 16) & 0xFF);
    (*out)[(*offset)++] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

}  // namespace

Response Perform(const std::string& url, esp_http_client_method_t method,
                 const std::vector<Header>& headers, const std::string& body, int timeout_ms,
                 const char* client_label)
{
    Response response = {};

    esp_http_client_config_t config = {};
    config.url = url.c_str();
    config.method = method;
    config.crt_bundle_attach = esp_crt_bundle_attach;
    config.timeout_ms = timeout_ms;
    config.event_handler = &EventHandler;
    config.user_data = &response;

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == nullptr) {
        response.error_code = "http_client_init_failed";
        response.error_message = std::string("Failed to initialize ") + client_label + " HTTP client";
        return response;
    }

    ApplyHeaders(client, headers);
    if (!body.empty()) {
        esp_http_client_set_post_field(client, body.c_str(), static_cast<int>(body.size()));
    }

    const esp_err_t err = esp_http_client_perform(client);
    response.status_code = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);

    if (err != ESP_OK) {
        response.error_code = "transport_error";
        response.error_message = esp_err_to_name(err);
    }
    return response;
}

Response PerformWavUpload(const std::string& url, const std::vector<Header>& headers,
                          const std::string& prefix, const recording_service::RecordedClip& clip,
                          const std::string& suffix, int timeout_ms, const char* client_label)
{
    Response response = {};

    esp_http_client_config_t config = {};
    config.url = url.c_str();
    config.method = HTTP_METHOD_POST;
    config.crt_bundle_attach = esp_crt_bundle_attach;
    config.timeout_ms = timeout_ms;

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == nullptr) {
        response.error_code = "http_client_init_failed";
        response.error_message =
            std::string("Failed to initialize ") + client_label + " upload client";
        return response;
    }

    ApplyHeaders(client, headers);

    const size_t total_bytes = prefix.size() + clip.wav_byte_count() + suffix.size();
    esp_err_t err = esp_http_client_open(client, static_cast<int>(total_bytes));
    if (err != ESP_OK) {
        response.error_code = "transport_error";
        response.error_message = esp_err_to_name(err);
        esp_http_client_cleanup(client);
        return response;
    }

    bool ok = true;
    const auto write_all = [&](const char* data, size_t len) {
        if (!ok || len == 0) {
            return;
        }
        const int written = esp_http_client_write(client, data, static_cast<int>(len));
        ok = written == static_cast<int>(len);
    };

    write_all(prefix.data(), prefix.size());
    const std::array<uint8_t, 44> header =
        BuildWavHeaderPcm16Mono(clip.sample_count(), clip.sample_rate_hz());
    write_all(reinterpret_cast<const char*>(header.data()), header.size());
    clip.ForEachChunk([&](const int16_t* chunk_data, size_t chunk_size) {
        if (chunk_data == nullptr || chunk_size == 0) {
            return;
        }
        write_all(reinterpret_cast<const char*>(chunk_data), chunk_size * sizeof(int16_t));
    });
    write_all(suffix.data(), suffix.size());

    if (!ok) {
        response.error_code = "transport_error";
        response.error_message = std::string("Failed streaming ") + client_label + " audio upload";
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return response;
    }

    const int response_length = esp_http_client_fetch_headers(client);
    response.status_code = esp_http_client_get_status_code(client);
    if (response.status_code <= 0 && response_length < 0) {
        response.error_code = "transport_error";
        response.error_message =
            std::string("Failed fetching ") + client_label + " upload response headers";
    } else {
        ReadBody(client, &response);
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return response;
}

std::string TrimCopy(std::string value)
{
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string TrimForLog(std::string value, size_t max_len)
{
    value = TrimCopy(std::move(value));
    if (value.size() <= max_len) {
        return value;
    }
    if (max_len <= 3) {
        return value.substr(0, max_len);
    }
    return value.substr(0, max_len - 3) + "...";
}

std::string JsonStringField(cJSON* root, const char* key)
{
    if (root == nullptr || key == nullptr) {
        return {};
    }
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsString(item) || item->valuestring == nullptr) {
        return {};
    }
    return item->valuestring;
}

std::string JsonNestedStringField(cJSON* root, const char* first, const char* second)
{
    if (root == nullptr) {
        return {};
    }
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root, first);
    if (!cJSON_IsObject(item)) {
        return {};
    }
    return JsonStringField(item, second);
}

std::string JsonToString(cJSON* root)
{
    if (root == nullptr) {
        return "{}";
    }
    char* raw = cJSON_PrintUnformatted(root);
    if (raw == nullptr) {
        return "{}";
    }
    std::string json(raw);
    cJSON_free(raw);
    return json;
}

std::array<uint8_t, 44> BuildWavHeaderPcm16Mono(size_t sample_count, uint32_t sample_rate_hz)
{
    constexpr uint16_t kChannels = 1;
    constexpr uint16_t kBitsPerSample = 16;
    constexpr uint16_t kBlockAlign = kChannels * (kBitsPerSample / 8U);
    const uint32_t data_bytes = static_cast<uint32_t>(sample_count * sizeof(int16_t));
    const uint32_t byte_rate = sample_rate_hz * kBlockAlign;

    std::array<uint8_t, 44> header = {};
    size_t offset = 0;
    std::memcpy(header.data() + offset, "RIFF", 4);
    offset += 4;
    AppendLe32(36U + data_bytes, &header, &offset);
    std::memcpy(header.data() + offset, "WAVEfmt ", 8);
    offset += 8;
    AppendLe32(16U, &header, &offset);
    AppendLe16(1U, &header, &offset);
    AppendLe16(kChannels, &header, &offset);
    AppendLe32(sample_rate_hz, &header, &offset);
    AppendLe32(byte_rate, &header, &offset);
    AppendLe16(kBlockAlign, &header, &offset);
    AppendLe16(kBitsPerSample, &header, &offset);
    std::memcpy(header.data() + offset, "data", 4);
    offset += 4;
    AppendLe32(data_bytes, &header, &offset);
    return header;
}

}  // namespace http
}  // namespace llm_service
