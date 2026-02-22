# Building a hybrid Python + Rust project with PyO3 and maturin

**The optimal approach in 2026 combines maturin 1.12 with PyO3 0.28, using a workspace layout that separates pure Rust crates from a thin PyO3 bindings crate, with Python source in a dedicated `python/` directory.** This pattern—proven by polars, pydantic-core, and dozens of high-performance Python packages—lets you ship a Python CLI that works standalone but accelerates dramatically when the Rust extension is installed. Below is a comprehensive guide covering project structure, configuration, the content-addressed store architecture, Ollama streaming integration, and Apple M3 Max specifics, with working code throughout.

---

## Recommended project structure and directory layout

The most maintainable layout for a multi-crate Rust workspace with Python bindings uses maturin's `python-source` directive to keep Python and Rust code cleanly separated. This mirrors how pydantic-core and polars organize their codebases:

```
my-project/
├── pyproject.toml                # Maturin build config + PEP 621 metadata
├── Cargo.toml                    # Workspace root
├── python/                       # Python source directory
│   └── my_project/
│       ├── __init__.py           # try/except import from _rust
│       ├── py.typed              # PEP 561 marker
│       ├── _rust.pyi             # Type stubs for IDE support
│       ├── _python_impl.py       # Pure Python fallback implementations
│       ├── cli.py                # Click CLI entry point
│       ├── models.py             # Pydantic models
│       └── utils.py              # Pure Python utilities
├── crates/
│   ├── core/                     # Pure Rust library (no PyO3)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── store.rs          # Content-addressed store
│   │       ├── ollama.rs         # Ollama streaming client
│   │       └── compress.rs       # zstd compression
│   └── python-bindings/          # PyO3 cdylib — thin wrapper
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs            # #[pymodule] entry point
└── tests/
    ├── test_python.py
    └── test_rust_integration.py
```

The key insight from pydantic-core's architecture is the **three-layer separation**: pure Python API surface, thin PyO3 translation layer, and core Rust logic with zero PyO3 dependencies. This lets you test and publish the Rust crates independently while keeping the bindings layer minimal.

**Root `Cargo.toml` (workspace):**
```toml
[workspace]
members = ["crates/*"]
resolver = "2"
```

**`crates/core/Cargo.toml`:**
```toml
[package]
name = "my-project-core"
version = "0.1.0"
edition = "2021"

[dependencies]
memmap2 = "0.9"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
zstd = "0.13"
blake3 = "1"
reqwest = { version = "0.12", features = ["json", "stream"] }
tokio = { version = "1", features = ["full"] }
futures-util = "0.3"
tokio-util = { version = "0.7", features = ["io"] }
```

**`crates/python-bindings/Cargo.toml`:**
```toml
[package]
name = "my-project-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["abi3-py310"] }
my-project-core = { path = "../core" }
```

---

## Maturin and PyO3 configuration for 2025–2026

**Maturin 1.12.0** (released February 2026) and **PyO3 0.28.2** form the current stable stack. Notable changes from earlier versions: PyO3 0.27+ auto-enables the `extension-module` feature (no longer needed in `[tool.maturin]`), `PyObject` is deprecated in favor of `Py<PyAny>`, and the new `PyUntypedBuffer` type enables type-erased buffer protocol access.

**Complete `pyproject.toml`:**
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "my-project"
dynamic = ["version"]
description = "Hybrid Python/Rust project with optional Rust acceleration"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "click>=8.1",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "mypy", "ruff"]

[project.scripts]
my-cli = "my_project.cli:main"

[tool.maturin]
python-source = "python"
module-name = "my_project._rust"
manifest-path = "crates/python-bindings/Cargo.toml"
strip = true
```

The `manifest-path` directive tells maturin which crate to build within the workspace. The `module-name` places the compiled extension at `my_project._rust`, avoiding namespace collisions with the Python package directory.

### The optional Rust backend pattern

The canonical approach uses a try/except import at the Python level. **Maturin does not natively support making Rust compilation optional**—instead, you structure the Python code to gracefully degrade:

```python
# python/my_project/__init__.py
try:
    from my_project._rust import (
        fast_store_get,
        fast_store_put,
        stream_chat_rust,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def store_get(hash_hex: str) -> bytes:
    if _HAS_RUST:
        return fast_store_get(hash_hex)
    return _python_fallback_get(hash_hex)

# Expose the flag for user introspection
def has_rust_backend() -> bool:
    return _HAS_RUST
```

Real-world projects like **cryptography** (which made Rust optional in v3.4 before mandating it in v3.5) and **hashsigs-py** use this exact pattern. For distribution, you can ship either a single package with Rust wheels for common platforms (users on unsupported platforms get the pure-Python fallback via sdist), or two separate packages where the Rust extension is an optional dependency.

---

## Exposing Rust functions to Python with PyO3

PyO3 0.28 provides automatic conversion for most Python types. The bindings crate should be a thin translation layer:

```rust
// crates/python-bindings/src/lib.rs
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use my_project_core::{ContentAddressedStore, OllamaClient};

#[pymodule]
#[pyo3(name = "_rust")]
fn my_rust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_store_put, m)?)?;
    m.add_function(wrap_pyfunction!(fast_store_get, m)?)?;
    m.add_class::<RustContentStore>()?;
    Ok(())
}

/// Return bytes to Python — PyBytes::new copies from Rust slice
#[pyfunction]
fn fast_store_get<'py>(py: Python<'py>, hash_hex: &str) -> PyResult<Bound<'py, PyBytes>> {
    let data = get_from_store(hash_hex)?;
    Ok(PyBytes::new(py, &data))
}

/// Accept bytes from Python — &[u8] is zero-copy from Python bytes
#[pyfunction]
fn fast_store_put(data: &[u8]) -> PyResult<String> {
    let hash = put_to_store(data)?;
    Ok(hex::encode(hash))
}

/// Return a dict with stats
#[pyfunction]
fn store_stats<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_blobs", 42u64)?;
    dict.set_item("total_bytes", 1_048_576u64)?;
    dict.set_item("segments", 3u32)?;
    Ok(dict)
}

/// Return Vec<T> — auto-converts to Python list
#[pyfunction]
fn list_hashes() -> Vec<String> {
    vec!["abcd1234".into(), "ef567890".into()]
}
```

### Handling pydantic models across the boundary

There is no direct Pydantic integration in PyO3. The most robust pattern uses `#[derive(FromPyObject)]` to extract attributes from any Python object, including Pydantic models:

```rust
#[derive(FromPyObject)]
struct ChatConfig {
    #[pyo3(attribute)]
    model: String,
    #[pyo3(attribute)]
    temperature: f64,
    #[pyo3(attribute)]
    max_tokens: u32,
}

#[pyfunction]
fn configure_chat(config: ChatConfig) -> PyResult<String> {
    Ok(format!("Model: {}, temp: {}", config.model, config.temperature))
}
```

For complex nested models, serialize via `.model_dump()` and extract with `#[pyo3(from_item_all)]`:

```rust
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct UserData {
    name: String,
    email: String,
    age: u32,
}

#[pyfunction]
fn process_user(model: &Bound<'_, PyAny>) -> PyResult<String> {
    let dict = model.call_method0("model_dump")?;
    let user: UserData = dict.extract()?;
    Ok(format!("User: {} ({})", user.name, user.email))
}
```

### Zero-copy data passing with the buffer protocol

For exposing memory-mapped data to Python as a `memoryview`, implement `__getbuffer__` on a `#[pyclass]`:

```rust
use pyo3::ffi;
use std::os::raw::c_int;
use std::sync::Arc;
use memmap2::Mmap;

#[pyclass]
struct MmapView {
    mmap: Arc<Mmap>,
    len: usize,
}

#[pymethods]
impl MmapView {
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err("read-only"));
        }
        unsafe {
            (*view).obj = ffi::_Py_NewRef(slf.as_ptr());
            (*view).buf = slf.mmap.as_ptr() as *mut _;
            (*view).len = slf.len as isize;
            (*view).itemsize = 1;
            (*view).readonly = 1;
            (*view).ndim = 1;
            (*view).format = c"B".as_ptr() as *mut _;
            (*view).shape = &mut (*view).len as *mut _ as *mut isize;
            (*view).strides = &mut (*view).itemsize as *mut _ as *mut isize;
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}
```

This lets Python code access memory-mapped segment data with `memoryview(mmap_view)` — true zero-copy for sealed, immutable segments. Use `Arc<Mmap>` to ensure the mapping outlives any Python references. For simpler cases, `PyBytes::new(py, &slice)` involves one copy but is fully safe.

---

## Content-addressed store with mmap and zstd

The store architecture uses **append-only segment files** mapped via memmap2, **BLAKE3** content hashing, **zstd dictionary compression** for similar blobs, and an in-memory index:

```rust
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

const SEGMENT_SIZE: usize = 64 * 1024 * 1024; // 64 MB segments

struct Segment {
    mmap: MmapMut,
    write_pos: usize,
    max_size: usize,
}

impl Segment {
    fn new(path: &PathBuf) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true).write(true).create(true)
            .open(path)?;
        file.set_len(SEGMENT_SIZE as u64)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self { mmap, write_pos: 0, max_size: SEGMENT_SIZE })
    }

    fn append(&mut self, data: &[u8]) -> Option<(usize, u32)> {
        let header = (data.len() as u32).to_le_bytes();
        let total = 4 + data.len();
        if self.write_pos + total > self.max_size { return None; }

        self.mmap[self.write_pos..self.write_pos + 4]
            .copy_from_slice(&header);
        self.mmap[self.write_pos + 4..self.write_pos + total]
            .copy_from_slice(data);
        let offset = self.write_pos;
        self.write_pos += total;
        Some((offset, data.len() as u32))
    }

    fn read(&self, offset: usize) -> &[u8] {
        let len = u32::from_le_bytes(
            self.mmap[offset..offset + 4].try_into().unwrap()
        ) as usize;
        &self.mmap[offset + 4..offset + 4 + len]
    }
}

struct ContentAddressedStore {
    active: Segment,
    sealed: Vec<Mmap>,        // Immutable, read-only mapped segments
    index: HashMap<[u8; 32], (u32, usize, u32)>, // hash → (seg_id, offset, len)
    compressor: zstd::bulk::Compressor<'static>,
    decompressor: zstd::bulk::Decompressor<'static>,
}

impl ContentAddressedStore {
    fn put(&mut self, raw: &[u8]) -> [u8; 32] {
        let hash = *blake3::hash(raw).as_bytes();
        if self.index.contains_key(&hash) { return hash; }

        let compressed = self.compressor.compress(raw).unwrap();
        match self.active.append(&compressed) {
            Some((offset, len)) => {
                self.index.insert(hash, (self.sealed.len() as u32, offset, len));
            }
            None => {
                // Seal current segment, create new one, retry
                self.rotate();
                let (offset, len) = self.active.append(&compressed).unwrap();
                self.index.insert(hash, (self.sealed.len() as u32, offset, len));
            }
        }
        hash
    }
}
```

**zstd dictionary compression** is particularly valuable for stores containing structurally similar JSON blobs. Train a dictionary from representative samples, then reuse it for **2–5× better compression ratios** on small documents:

```rust
let samples: Vec<&[u8]> = existing_blobs.iter().map(|b| b.as_slice()).collect();
let dict_data = zstd::dict::from_samples(&samples, 64 * 1024)?; // 64KB dict
let encoder_dict = zstd::dict::EncoderDictionary::copy(&dict_data, 3);
let decoder_dict = zstd::dict::DecoderDictionary::copy(&dict_data);
```

For **streaming NDJSON ingestion** into the store, serde_json's `StreamDeserializer` parses directly from memory-mapped bytes with zero allocation for the input buffer:

```rust
use serde_json::Deserializer;

fn ingest_ndjson(mmap: &[u8], store: &mut ContentAddressedStore) {
    let stream = Deserializer::from_slice(mmap).into_iter::<serde_json::Value>();
    for result in stream {
        let value = result.expect("valid JSON");
        let serialized = serde_json::to_vec(&value).unwrap();
        store.put(&serialized);
    }
}
```

---

## Streaming from Ollama's REST API in Rust

Ollama streams `/api/chat` responses as **newline-delimited JSON** (`application/x-ndjson`). Each line is a self-contained JSON object. Tool calls arrive as a single atomic chunk with `message.tool_calls` populated and empty `content`. The final chunk has `done: true` with generation statistics.

The recommended approach uses `tokio_util::io::StreamReader` to convert reqwest's byte stream into line-by-line async reading:

```rust
use futures_util::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    model: String,
    message: StreamMessage,
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StreamMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ToolCall {
    function: ToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ToolCallFunction {
    name: String,
    arguments: HashMap<String, serde_json::Value>,
}

async fn stream_chat(
    base_url: &str,
    request: &serde_json::Value,
) -> Result<(String, Vec<ToolCall>), Box<dyn std::error::Error>> {
    use tokio::io::AsyncBufReadExt;
    use tokio_util::io::StreamReader;

    let response = reqwest::Client::new()
        .post(format!("{}/api/chat", base_url))
        .json(request)
        .send()
        .await?;

    let byte_stream = response.bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    let reader = StreamReader::new(byte_stream);
    let mut lines = reader.lines();

    let mut content = String::new();
    let mut tool_calls = Vec::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() { continue; }
        let chunk: ChatStreamChunk = serde_json::from_str(&line)?;

        if !chunk.message.content.is_empty() {
            content.push_str(&chunk.message.content);
        }
        if let Some(calls) = chunk.message.tool_calls {
            tool_calls.extend(calls);
        }
        if chunk.done { break; }
    }

    Ok((content, tool_calls))
}
```

**Critical implementation detail**: TCP chunks from reqwest do not align with JSON line boundaries. The `StreamReader` + `lines()` approach handles this correctly by buffering internally. For manual implementations, maintain a byte buffer and split on `\n` boundaries. Tool call chunks are atomic—the entire `tool_calls` array arrives in a single NDJSON line.

For production use, the **`ollama-rs`** crate (v0.3.4, ~21K monthly downloads) provides a higher-level API with streaming, tool calling, and chat history support. Use the raw reqwest approach only when you need fine-grained control over streaming behavior.

---

## Apple M3 Max specific considerations

Three critical differences on Apple Silicon affect this project directly.

**16KB page size** is the most impactful. Apple Silicon uses 16KB memory pages versus 4KB on x86_64. Any `mmap` offset must be a multiple of **16,384 bytes**, not 4,096. Code that hardcodes 4KB alignment will silently work on Intel Macs but fail with `EINVAL` on Apple Silicon. Query the page size at runtime:

```rust
let page_size = page_size::get(); // 16384 on Apple Silicon
let aligned_offset = (desired_offset / page_size) * page_size;
let mmap = unsafe {
    MmapOptions::new().offset(aligned_offset as u64).map(&file)?
};
```

For the segment file design, use segment sizes that are multiples of 16KB (the 64MB default is fine). The memmap2 crate handles alignment transparently for full-file mappings, but partial mappings with explicit offsets require attention.

**Unified memory architecture** means all **400 GB/s** of memory bandwidth is shared between CPU and GPU—no discrete VRAM. This eliminates NUMA concerns entirely: all CPU cores have uniform memory access latency. For a content-addressed store doing large sequential reads, this bandwidth is excellent. However, GPU-intensive tasks (display compositing, local ML inference via Ollama) compete for the same bandwidth. Monitor memory *pressure*, not just usage, since macOS aggressively compresses unused pages before swapping.

**Maturin build configuration** for Apple Silicon requires macOS 11.0+ as the deployment target:

```bash
# Native ARM64 build (verify with: rustc -vV | grep host)
maturin build --release

# Universal binary for both architectures
rustup target add aarch64-apple-darwin x86_64-apple-darwin
maturin build --release --target universal2-apple-darwin
```

In `.cargo/config.toml`, the default target CPU for `aarch64-apple-darwin` is `apple-m1` since Rust 1.71. Avoid using `-Ctarget-cpu=native`—it can paradoxically select fewer optimizations than the default due to an LLVM detection issue. NEON SIMD, hardware AES, and SHA-2 are enabled by default on all Apple Silicon targets.

**Release profile for M3 Max performance** (`Cargo.toml`):
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

---

## How major projects structure their Python/Rust boundary

The ecosystem has converged on a clear set of patterns. **Polars** uses a monorepo with ~20 pure Rust crates in `crates/` and a thin Python wrapper in `py-polars/`—Python is described as a "thin wrapper" with all computation happening in Rust. **Pydantic-core** follows a three-layer architecture: Python schema definitions (TypedDicts) → Rust validation engine (`SchemaValidator`, `CombinedValidator` enum dispatch) → `.pyi` stub files for IDE support. Both use maturin as the build backend with the `python-source` directive.

**Ruff** takes a different approach entirely: it's a pure Rust binary distributed as a Python package via maturin—no PyO3 bindings at all, just a CLI tool installed via `pip`. **Cryptography** uses setuptools-rust (not maturin) and made Rust optional in v3.4 before mandating it in v3.5, citing memory safety.

The key architectural lessons from these projects:

- **Most mature projects make Rust mandatory**, not optional—the performance *is* the product. The optional pattern works best during a transition period.
- **Type stubs (`.pyi` files) are essential** for IDE support when the core is Rust. Pydantic-core's `_pydantic_core.pyi` is a good reference.
- **The bindings crate should be thin**—put business logic in pure Rust crates without PyO3 dependencies. This enables independent testing, benchmarking, and potential crates.io publication.
- **Profile-guided optimization (PGO)** is used by pydantic-core for production builds: compile, run benchmarks to generate profiles, recompile with profile data. This yields measurable gains for hot-path code.

---

## Conclusion

The hybrid Python + Rust stack in 2026 is mature and well-supported. Maturin 1.12's `manifest-path` directive handles workspace builds cleanly, and PyO3 0.28's automatic type conversions reduce boilerplate significantly. The most important architectural decision is separating pure Rust crates from the PyO3 bindings layer—this mirrors pydantic-core's proven three-layer design and keeps your core logic testable without Python in the loop.

For the content-addressed store, the combination of memmap2 + blake3 + zstd dictionary compression handles the storage layer efficiently, with sealed segments safely exposed to Python via the buffer protocol. For Ollama streaming, the `StreamReader` + `lines()` pattern correctly handles TCP chunk boundaries in NDJSON streams. On Apple Silicon, the **16KB page size** is the single most important gotcha—design segment offsets around it from the start rather than retrofitting later. The unified memory architecture is otherwise a pure advantage for memory-mapped workloads, delivering 400 GB/s bandwidth without NUMA complexity.
