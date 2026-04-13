#!/bin/bash

# Script para verificar recursos reales en deimos y phobos
# Usa qstat y SSH para obtener información actual del cluster

echo "=============================================================================="
echo "  VERIFICACIÓN ACTUAL DE RECURSOS EN CLUSTER"
echo "=============================================================================="
echo ""

# Función para verificar un nodo
check_node() {
    local node=$1
    echo "🔍 Consultando $node..."
    echo ""
    
    # Intenta con qstat
    echo "  [qstat]"
    qstat -f -h "$node" 2>/dev/null | head -20 || echo "  (qstat no disponible o nodo no responde)"
    
    # Intenta con SSH
    echo ""
    echo "  [nvidia-smi vía SSH]"
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader" 2>/dev/null; then
        echo "  ✓ SSH conexión exitosa"
    else
        echo "  (SSH timeout - intenta manualmente: ssh $node 'nvidia-smi')"
    fi
    
    echo ""
}

# Verificar deimos
check_node "deimos"

# Verificar phobos
check_node "phobos"

echo "=============================================================================="
echo "  COMANDO PARA EJECUTAR MANUALMENTE"
echo "=============================================================================="
echo ""
echo "Si los comandos anteriores no funcionan, intenta directamente:"
echo ""
echo "ssh deimos 'nvidia-smi'"
echo "ssh phobos 'nvidia-smi'"
echo ""
echo "Para información más detallada:"
echo ""
echo "ssh deimos 'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader'"
echo "ssh phobos 'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader'"
echo ""

echo "=============================================================================="
echo "  INFORMACIÓN CONOCIDA"
echo "=============================================================================="
echo ""
echo "Cola student.q:"
echo "  ✓ pcgtx1080: GTX 1080 (8GB, CC 6.1)"
echo "  ✓ pcgtx1070: GTX 1070 (8GB, CC 6.1)"
echo "  ✓ pcgtx970:  GTX 970  (4GB, CC 5.2)"
echo ""
echo "Cola tfm.q:"
echo "  ? deimos:    DESCONOCIDO (probablemente GPU moderna)"
echo "  ? phobos:    DESCONOCIDO (probablemente GPU moderna)"
echo ""
