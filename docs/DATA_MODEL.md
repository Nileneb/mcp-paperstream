# PaperStream Data Model

## Hierarchy: Text → Embedding → Chunk → Voxels

```
┌─────────────────────────────────────────────────────────────────┐
│  TEXT (Papers & Rules)                                          │
│  - Papers: PDF sections (abstract, methods, results, etc.)      │
│  - Rules: Positive/negative phrase lists                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ BioBERT (768-dim)
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDING (768 dimensions)                                      │
│  - Semantic vector representation                                │
│  - Base64 encoded for transport: embedding_b64                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ Container
┌─────────────────────────────────────────────────────────────────┐
│  CHUNK (Unity: INVISIBLE cube at position)                       │
│  - Container for embedding + voxels                              │
│  - Paper: N chunks (one per section)                             │
│  - Rule: 2 chunks (positive=green, negative=red)                 │
│  - Position defines local origin (0,0,0) for voxels inside       │
│  - connects_to: Array of chunk_ids for wire/lane connections     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ Reshape 768 → 8x8x12
┌─────────────────────────────────────────────────────────────────┐
│  VOXELS (Unity: VISIBLE cubes for interaction)                   │
│  - 8x8x12 grid = 768 values (matches embedding dim)              │
│  - Each voxel = one embedding dimension visualized               │
│  - Rendered relative to chunk's position                         │
│  - voxels: {grid_size: [8,8,12], voxels: [[x,y,z,value],...]}   │
└─────────────────────────────────────────────────────────────────┘
```

## Unified Chunk Structure

**BOTH Papers AND Rules use the SAME structure!**

```json
{
  "chunk_id": 0,
  "chunk_type": "abstract | methods | results | positive | negative",
  "text_preview": "First 500 chars of source text...",
  "embedding_b64": "base64-encoded-768-float32-array",
  "voxels": {
    "grid_size": [8, 8, 12],
    "voxels": [[x, y, z, value], ...],
    "voxel_count": 767,
    "fill_ratio": 0.999
  },
  "color": {"r": 0.2, "g": 0.9, "b": 0.3},
  "position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "connects_to": [1, 2]
}
```

## Paper Molecule (Chain Layout)

```
[Abstract] ─► [Intro] ─► [Methods] ─► [Results] ─► [Discussion]
   Blue       Green      Yellow        Red         Purple
   
Chunk 0     Chunk 1     Chunk 2      Chunk 3      Chunk 4
x=0.0       x=2.0       x=4.0        x=6.0        x=8.0
```

- **Layout**: `chain` (sequential)
- **Connection Type**: `sequential` (one after another)
- **Colors**: Section-specific (see SECTION_COLORS)

## Rule Molecule (Dipole Layout)

```
[POSITIVE] ◄────────► [NEGATIVE]
   Green                 Red
   
 Chunk 0               Chunk 1
 x=0.0                 x=2.0
```

- **Layout**: `dipole` (two poles)
- **Connection Type**: `polar` (opposing forces)
- **Colors**: Green (positive), Red (negative)

## Unity Rendering Flow

```csharp
// 1. Create molecule container
var moleculeRoot = new GameObject(molecule.molecule_id);

// 2. Create each chunk (INVISIBLE)
foreach (var chunk in molecule.chunks) {
    var chunkObj = new GameObject($"Chunk_{chunk.chunk_id}");
    chunkObj.transform.position = new Vector3(
        chunk.position.x, 
        chunk.position.y, 
        chunk.position.z
    );
    chunkObj.transform.parent = moleculeRoot.transform;
    
    // 3. Create voxels (VISIBLE) inside chunk
    foreach (var voxel in chunk.voxels.voxels) {
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.parent = chunkObj.transform;
        cube.transform.localPosition = new Vector3(
            voxel[0] * 0.1f,  // Scale factor
            voxel[1] * 0.1f,
            voxel[2] * 0.1f
        );
        cube.transform.localScale = Vector3.one * 0.08f;
        
        // Color based on chunk color + voxel intensity
        var renderer = cube.GetComponent<Renderer>();
        var color = new Color(chunk.color.r, chunk.color.g, chunk.color.b);
        color *= voxel[3]; // Multiply by intensity
        renderer.material.color = color;
    }
    
    // 4. Draw WIRES between voxels INSIDE this chunk
    // Wires connect voxels within the same chunk to show embedding structure
    DrawVoxelWires(chunkObj, chunk.voxels.voxels);
}
```

## Wires/Lanes (WICHTIG!)

**Wires verbinden VOXELS innerhalb eines Chunks - NICHT Chunks untereinander!**

```
┌─────────────────────────────────────────────────────────────────┐
│  CHUNK (INVISIBLE CONTAINER)                                     │
│                                                                  │
│   Voxel[0,0,0] ──Wire── Voxel[1,0,0] ──Wire── Voxel[2,0,0]      │
│        │                     │                     │             │
│       Wire                  Wire                  Wire           │
│        │                     │                     │             │
│   Voxel[0,1,0] ──Wire── Voxel[1,1,0] ──Wire── Voxel[2,1,0]      │
│        │                     │                     │             │
│       ...                   ...                   ...            │
│                                                                  │
│   8x8x12 Voxels verbunden durch Wires = Grid-Struktur           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Wire Purpose
- **Visualize embedding structure**: 768 dimensions as 3D grid
- **Show relationships**: Which voxels belong together
- **Aid navigation**: Help user understand the embedding topology
- **Group voxels**: Multiple embeddings in one chunk → wires show grouping

### Wire Implementation (C#)

```csharp
// Draw wires BETWEEN VOXELS within a chunk (not between chunks!)
void DrawVoxelWires(Transform chunkParent, List<int[]> voxels) {
    var wireContainer = new GameObject("Wires");
    wireContainer.transform.parent = chunkParent;
    
    // Connect adjacent voxels to show grid structure
    foreach (var voxel in voxels) {
        int x = voxel[0], y = voxel[1], z = voxel[2];
        
        // Connect to neighbors in X direction
        if (HasVoxel(x+1, y, z)) {
            DrawWire(voxel, GetVoxel(x+1, y, z), wireContainer);
        }
        // Connect to neighbors in Y direction
        if (HasVoxel(x, y+1, z)) {
            DrawWire(voxel, GetVoxel(x, y+1, z), wireContainer);
        }
        // Connect to neighbors in Z direction
        if (HasVoxel(x, y, z+1)) {
            DrawWire(voxel, GetVoxel(x, y, z+1), wireContainer);
        }
    }
}

void DrawWire(int[] from, int[] to, Transform parent) {
    var lineRenderer = new GameObject("Wire").AddComponent<LineRenderer>();
    lineRenderer.transform.parent = parent;
    lineRenderer.positionCount = 2;
    lineRenderer.startWidth = 0.01f;
    lineRenderer.endWidth = 0.01f;
    lineRenderer.SetPosition(0, new Vector3(from[0], from[1], from[2]) * 0.1f);
    lineRenderer.SetPosition(1, new Vector3(to[0], to[1], to[2]) * 0.1f);
}
```

## Section Colors

| Type         | RGB                | Hex       |
|--------------|-------------------|-----------|
| abstract     | (0.2, 0.6, 0.9)   | #3399E6   |
| introduction | (0.3, 0.8, 0.3)   | #4DCC4D   |
| methods      | (0.9, 0.7, 0.2)   | #E6B333   |
| results      | (0.8, 0.3, 0.3)   | #CC4D4D   |
| discussion   | (0.7, 0.4, 0.9)   | #B366E6   |
| conclusion   | (0.5, 0.5, 0.5)   | #808080   |
| positive     | (0.2, 0.9, 0.3)   | #33E64D   |
| negative     | (0.9, 0.2, 0.2)   | #E63333   |

## API Endpoints

### Paper Chunks
```http
GET /api/papers/{paper_id}/chunks
GET /api/chunks?paper_id=DOI:10.1234/example
```

### Rule Chunks
```http
GET /api/rules/{rule_id}/chunks
GET /api/rules/chunks  # All rules
GET /api/rule-chunks?rule_id=is_rct
```

### Jobs (includes chunks)
```http
GET /api/jobs/next?device_id=unity_game
```

Returns job with `chunks` array in same format.

## Embedding Decode (C#)

```csharp
public float[] DecodeEmbedding(string b64) {
    byte[] bytes = Convert.FromBase64String(b64);
    float[] embedding = new float[768];
    Buffer.BlockCopy(bytes, 0, embedding, 0, bytes.Length);
    return embedding;
}
```

## Voxel Grid Decode (C#)

```csharp
public void SpawnVoxels(JObject chunk, Transform parent) {
    var voxels = chunk["voxels"]["voxels"] as JArray;
    var color = new Color(
        chunk["color"]["r"].Value<float>(),
        chunk["color"]["g"].Value<float>(),
        chunk["color"]["b"].Value<float>()
    );
    
    foreach (JArray voxel in voxels) {
        int x = voxel[0].Value<int>();
        int y = voxel[1].Value<int>();
        int z = voxel[2].Value<int>();
        float intensity = voxel[3].Value<float>();
        
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.parent = parent;
        cube.transform.localPosition = new Vector3(x, y, z) * 0.1f;
        cube.transform.localScale = Vector3.one * 0.08f;
        cube.GetComponent<Renderer>().material.color = color * intensity;
    }
}
```
