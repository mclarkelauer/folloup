# LLM Service

`components/llm_service` is the only part of the firmware that talks to an AI provider. It owns
the provider choice, one API key per provider, the readiness state the UI gates on, the
background credential check, and the setup-portal routes. Two providers are implemented behind a
private backend interface:

| Provider | Id | Text / summaries | Speech-to-text | Token counting |
| --- | --- | --- | --- | --- |
| Muse (Meta Model API, default) | `muse` | `POST https://api.meta.ai/v1/chat/completions`, model `muse-spark-1.3` (`CONFIG_FOLLOWUP_MUSE_TEXT_MODEL`) | `POST https://api.meta.ai/v1/asr/transcribe`, multipart, model `muse-voice-transcribe-1.0` | none; callers use the local estimate |
| Gemini (Google Generative Language API) | `gemini` | `:generateContent`, model `models/gemini-2.5-flash-lite` (`CONFIG_FOLLOWUP_GEMINI_MODEL`) | resumable file upload + `:generateContent` with `fileData` | `:countTokens` |

Both providers authenticate with a cheap model `GET` (`/v1/models/{model}` on Meta,
`/v1beta/models/{model}` on Google) whenever the network comes up or a key changes. The
`runtime.ready` flag is true only when the active provider has a key **and** that probe
succeeded; `transcription_service`, `summary_service`, `recording_session_service`, the status
bar, and the summarize page all gate on it.

## Internal Layout

| File | Role |
| --- | --- |
| `include/llm_service.h` | Public API and snapshot shapes (namespace `llm_service`). |
| `llm_service.cpp` | Settings (NVS + Kconfig), snapshot/event plumbing, auth task, portal routes, dispatch to the active backend. |
| `llm_backend.h` | Private `backend::Backend` interface: `Authenticate`, `GenerateText`, `CountTokens`, `Transcribe`. |
| `llm_http.{h,cpp}` | Shared `esp_http_client` wrappers (`Perform`, `PerformWavUpload`), cJSON helpers, WAV header builder. |
| `muse_backend.cpp` | Meta Model API adapter. |
| `gemini_backend.cpp` | Google Generative Language API adapter. |

`PerformWavUpload` streams the clip straight from the recording chunks (44-byte WAV header +
PCM16) between a caller-supplied prefix and suffix, so the Muse multipart body and the Gemini raw
upload share one code path and never buffer the WAV in RAM.

## Settings And Key Sources

NVS namespace `llm`:

| Key | Meaning |
| --- | --- |
| `provider` | Active provider id (`muse` / `gemini`). Absent → `CONFIG_FOLLOWUP_LLM_PROVIDER` (default `muse`). |
| `key_muse` | Stored Muse key. |
| `key_gemini` | Stored Gemini key. On first boot after upgrading from the Gemini-only firmware, the legacy `gemini/api_key` value is imported here. |

For each provider the effective key is the stored NVS key, else the built-in
`CONFIG_FOLLOWUP_MUSE_API_KEY` / `CONFIG_FOLLOWUP_GEMINI_API_KEY`. The snapshot reports the
source as `nvs`, `sdkconfig`, or `none`.

## Portal Endpoints

| Method | Path | Body | Effect |
| --- | --- | --- | --- |
| `GET` | `/api/settings/llm` | — | Full snapshot. |
| `PATCH` | `/api/settings/llm` | `{"provider":"muse"}` and/or `{"api_key":"...","key_provider":"muse"}` | Switch the active provider and/or store a key. `key_provider` defaults to the active provider. Any change re-runs authentication. |
| `POST` | `/api/settings/llm/reset` | optional `{"provider":"gemini"}` | Clear that provider's stored key (default: the active one). |
| `GET` | `/api/runtime/llm` | — | Same snapshot (kept for symmetry with the other runtime routes). |

Snapshot response shape:

```json
{
  "success": true,
  "message": "Muse API key stored",
  "settings": {
    "provider": "muse",
    "provider_display_name": "Muse",
    "configured": true,
    "has_key": true,
    "last4": "ab12",
    "has_stored_api_key": true,
    "has_sdkconfig_api_key": false,
    "api_key_source": "nvs",
    "api_key_last4": "ab12",
    "model_name": "muse-spark-1.3",
    "transcription_model_name": "muse-voice-transcribe-1.0",
    "providers": {
      "gemini": { "display_name": "Gemini", "has_key": false, "last4": "", "api_key_source": "none",
                  "model_name": "models/gemini-2.5-flash-lite",
                  "transcription_model_name": "models/gemini-2.5-flash-lite" },
      "muse":   { "display_name": "Muse", "has_key": true, "last4": "ab12", "api_key_source": "nvs",
                  "model_name": "muse-spark-1.3", "transcription_model_name": "muse-voice-transcribe-1.0" }
    }
  },
  "runtime": {
    "initialized": true, "ready": true, "request_in_flight": false,
    "auth_checked": true, "authenticated": true,
    "last_http_status": 200, "last_status_message": "Authenticated with muse-spark-1.3",
    "last_model_resource_name": "muse-spark-1.3", "last_model_display_name": "muse-spark-1.3",
    "last_error_code": "", "last_error_message": ""
  }
}
```

Errors return `{"success":false,"message":...,"error_code":...,"field":...}` with HTTP 400 for
validation problems and 500 for NVS failures.

## Error Codes Surfaced To The UI

Backend results carry the provider's own error code: Gemini's `error.status` (for example
`RESOURCE_EXHAUSTED`) or Meta's OpenAI-style `error.code` (for example `invalid_api_key`,
`rate_limit_exceeded`). `llm_service::IsQuotaErrorCode` recognises the quota / rate-limit codes
of both so the app shell can show a "quota exceeded" toast regardless of provider.

## Muse Transcription Request

The clip is sent exactly as recorded (16 kHz, mono, 16-bit PCM WAV), which Meta accepts
directly. The multipart body is:

```
--<boundary>
Content-Disposition: form-data; name="request"
Content-Type: application/json

{"mode":"DIARIZATION","model":"muse-voice-transcribe-1.0","audioEncoding":"WAV"}
--<boundary>
Content-Disposition: form-data; name="audio"; filename="clip.wav"
Content-Type: audio/wav

<WAV bytes>
--<boundary>--
```

The response's top-level `transcript` is used; if it is empty the per-speaker `turns[].transcript`
values are joined instead.

## Setup Portal

The portal's "AI Provider" card (`webserver/src/portal/llmProvider.ts`) shows a Muse/Gemini
picker and the API key for the picked provider. Changing the picker PATCHes `provider`; saving a
key PATCHes `api_key` + `key_provider` and also makes that provider active; Clear POSTs the reset
route for the picked provider. After any change the portal re-renders from the returned
`settings.providers` map, so each provider's masked key stays visible when you switch.
