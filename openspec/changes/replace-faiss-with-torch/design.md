## Context

Actualmente `HippoRAG2Constructor` usa FAISS (`faiss-gpu-cu12`) exclusivamente para tres operaciones básicas de álgebra lineal:

1. `faiss.normalize_L2(emb)` — normalizar vectores L2
2. `faiss.IndexFlatIP(dim)` — índice flat de producto interno (fuerza bruta exacta)
3. `index.search(query, k)` — búsqueda top-k por producto interno

Estas tres operaciones tienen equivalentes directos en `torch`, que ya es dependencia del proyecto. El `uv.lock` actual contiene `torch==2.8.0` y `faiss-gpu-cu12==1.14.1.post1`, este último sin wheel para `aarch64`.

La máquina actual es `aarch64` con NVIDIA GB10, CUDA 13.0 y driver 580.142. `torch>=2.9.0` tiene soporte CUDA nativo en `aarch64` a través de los nuevos paquetes `nvidia-*`.

## Goals / Non-Goals

**Goals:**
- Eliminar la dependencia de `faiss-gpu-cu12` del proyecto
- Implementar las 3 operaciones FAISS con torch (`IndexFlatIP` → `torch.mm`, `normalize_L2` → `F.normalize`, `search` → `torch.topk`)
- Subir `torch` a `>=2.9.0` para habilitar CUDA en `aarch64`
- Mantener exactamente el mismo comportamiento numérico (mismos resultados de búsqueda)
- Pasar todos los tests existentes

**Non-Goals:**
- No cambiar el algoritmo de búsqueda (sigue fuerza bruta exacta)
- No añadir índices aproximados o cuantización
- No modificar otros constructores ni módulos del proyecto
- No tocar la API pública de `HippoRAG2Constructor`

## Decisions

### 1. Usar `torch` en lugar de `numpy` puro

**Alternativas consideradas:**
- `numpy` puro: viñetas `np.dot`, `np.argsort`, `np.linalg.norm`. Misma complejidad de código, pero sin beneficio GPU.
- `faiss-cpu`: cero cambios de código, pero mantiene una dependencia de ~50 MB que no aporta valor sobre torch.

**Decisión:** `torch` porque los embeddings ya vienen como tensores (`torch.Tensor`) desde `self.text_emb_model.encode()`. Usar torch evita conversiones `tensor → numpy → tensor` y se beneficia de GPU automáticamente si está disponible.

### 2. Subir torch a `>=2.9.0` (efectivamente 2.11.0)

**Alternativas consideradas:**
- Mantener `>=2.4.1` y forzar `torch==2.8.0`: no hay CUDA en aarch64 en 2.8.0.
- Subir solo a `>=2.9.0`: uv resolvería a 2.11.0 de todas formas.

**Decisión:** `>=2.9.0` es suficiente como cota inferior (es cuando se reorganizaron los paquetes `nvidia-*`). uv resolverá a 2.11.0 que es la última estable.

### 3. Reemplazo 1:1 de las 3 operaciones FAISS

| FAISS                         | Torch equivalente                          |
|-------------------------------|---------------------------------------------|
| `faiss.normalize_L2(x)`       | `F.normalize(x, p=2, dim=1)`               |
| `faiss.IndexFlatIP(dim)`      | `torch.empty(0, dim)` como placeholder      |
| `index.add(embeddings)`       | Concatenar en tensor: `torch.cat(...)`      |
| `index.search(query, k)`      | `torch.mm(query, idx.T)` + `torch.topk(k)` |
| `index.ntotal`                | `index_tensor.shape[0]`                     |

### 4. Cambio de tipo en atributos de la clase `HippoRAG2Constructor`

Actualmente:
```python
self.node_indices_by_type: dict[str, faiss.IndexFlatIP] = {}
self.fact_index: faiss.IndexFlatIP | None = None
```

Cambiarán a:
```python
self.node_indices_by_type: dict[str, torch.Tensor] = {}
self.fact_index: torch.Tensor | None = None
```

### 5. Migración incremental: primero torch, luego eliminar FAISS

**Decisión:** Hacer ambos cambios en un solo PR porque son atómicos. Si se sube torch sin eliminar FAISS, el lock no resuelve. Si se elimina FAISS sin implementar torch, el código no compila. Un solo commit con ambos cambios.

## Risks / Trade-offs

- **[Riesgo] Diferencias numéricas mínimas**: torch y FAISS pueden diferir en la última cifra decimal por orden de operaciones → **Mitigación**: los tests pasan con tolerancia `atol=1e-5`. La búsqueda top-k es robusta a estas diferencias.
- **[Riesgo] Consumo de memoria**: cargar el índice como `torch.Tensor` en GPU puede aumentar uso de VRAM → **Mitigación**: el índice se mantiene en CPU por defecto (`index.add` de FAISS también es CPU). Solo se mueve a GPU si el usuario explícitamente lo desea.
- **[Riesgo] Regresión en torch>=2.9.0**: cambios de API entre 2.4.1 y 2.11.0 podrían romper otras partes del código → **Mitigación**: ejecutar todos los tests tras la migración.

## Open Questions

- ¿Hay tests que usen específicamente atributos de `faiss.IndexFlatIP` (como `index.ntotal`) y necesiten actualización? → Se revisará durante la implementación.
