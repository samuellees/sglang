# PCIe AllReduce Configs

Default policy files are stored as:

```text
<device_name>/tp<TP>/h<HIDDEN>/<dtype>/policy.json
```

Example:

```text
NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition/tp2/h2048/bf16/policy.json
```

Set `SGLANG_PCIE_AR_CONFIG_DIR` to use an external tuning cache.
