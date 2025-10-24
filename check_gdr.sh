#!/bin/bash
echo "=== GDR Environment Check ==="
echo ""

echo "1. GDRCopy kernel module:"
lsmod | grep gdrdrv || echo "  NOT LOADED"
echo ""

echo "2. GDRCopy device:"
ls -l /dev/gdrdrv 2>/dev/null || echo "  NOT FOUND"
echo ""

echo "3. GDRCopy library:"
ldconfig -p | grep libgdrapi || echo "  NOT FOUND in ldconfig"
find /usr -name "libgdrapi*" 2>/dev/null
echo ""

echo "4. CUDA version:"
nvcc --version 2>/dev/null || echo "  nvcc not found"
nvidia-smi | head -n 5
echo ""

echo "5. Libfabric version and providers:"
fi_info -l 2>/dev/null || echo "  fi_info not found"
echo ""

echo "6. Check for CXI provider with HMEM support:"
fi_info -p cxi 2>/dev/null | grep -i hmem || echo "  No HMEM support found in CXI provider"
echo ""

echo "7. AWS OFI NCCL plugin:"
find /usr -name "*nccl*ofi*" 2>/dev/null
echo ""

echo "8. Check plugin build info:"
strings $(find /usr -name "libnccl-net.so" 2>/dev/null | head -1) 2>/dev/null | grep -i "gdr\|hmem" || echo "  No GDR/HMEM strings found"
echo ""

echo "=== Recommendations ==="
echo "If GDRCopy device exists but GDR is disabled, the AWS OFI plugin likely wasn't"
echo "compiled with GDRCopy support. You need to rebuild it or use a pre-built version"
echo "that includes GDRCopy support."
