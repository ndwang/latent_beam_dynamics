#!/bin/bash
# Track one sample directory with Tao
cd "$1" && tao -init_file /pscratch/sd/n/ndwang/latent_beam_dynamics/tao.init \
    -noplot -lat lattice.bmad -beam_init_position_file beam.h5 <<'EOF'
set global track_type = beam
quit
EOF
