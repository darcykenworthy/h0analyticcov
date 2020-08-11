for i in "$1"/lowz*.pickle; do
    [ -f "$i" ] || break
    sbatch marginal.sbatch $i
done
