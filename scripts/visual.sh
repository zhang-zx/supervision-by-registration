python ./exps/visual.py --meta snapshots/CPM-SBR-mobile/metas/ --model eval-epoch-016-050-01-02.pth --save cache_data/cache/mobile
ffmpeg -start_number 3 -i cache_data/cache/mobile/image%04d.png -b:v 30000k -vf "fps=30" -pix_fmt yuv420p cache_data/cache/mobile.mp4
cp cache_data/cache/mobile.mp4 ../.jupyter