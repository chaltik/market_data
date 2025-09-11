tmpfile="$(mktemp -t ndx_today_XXXX).yaml"
python update_ndx_history.py
PYTHONPATH=. python gen_ndx_yaml.py --mode asof --asof "$(date +%F)" --output "$tmpfile"
eval $(cat .env) PYTHONPATH=. python data_retrieval.py --assets_file "$tmpfile"
rm -f "$tmpfile"