find ./**/actor_0/ -name "progress.json" | while read line; do
    echo "Fixing " $line
    mv $line $line.bak
    python3 ./fix_time_lag.py < $line.bak > $line
done
