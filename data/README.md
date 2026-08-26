# Local data

Raw datasets are excluded from Git by default. Before publishing any file from
this directory, record its source, license, consent or privacy constraints,
checksum, and preprocessing status.

Prefer reproducible download or preparation scripts over committing audio and
other large binary files.

## MeowAgeNet

The official cropped-call audio is reconstructed from the pinned upstream
commit by running:

```powershell
node .\scripts\build_meowagenet_manifest.mjs
```

The ignored local files are stored under
`data/meowagenet/official-3d02295bef15/AudioCropped`. Version-controlled source,
manifest, checksum, and cat-ID metadata are stored under
`metadata/datasets/meowagenet`.

Use `--verify-only` to re-hash an existing local copy without downloading
missing or invalid files.
