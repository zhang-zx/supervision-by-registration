python ./exps/visual.py --meta snapshots/CPM-SBR-small/metas/ --model eval-epoch-049-050-00-01.pth --save cache_data/cache/small
ffmpeg -start_number 3 -i cache_data/cache/small/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/small.mp4
cp cache_data/cache/small.mp4 ../.jupyter