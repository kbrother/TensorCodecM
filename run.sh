#python main.py train -p ../data/flowers/flowers -d flowers -de 0 1 2 3 -rk 16 -hs 17 -sp results/flowers_r16_h17_sr0.05 -e 500 -lr 1e-1 -b 8388608 -sr 0.05
#python main.py train -p ../data/hsv/hsv -d hsv -de 0 1 2 3 -rk 10 -hs 8 -sp results/hsv_r10_h8_sr0.05 -e 500 -lr 1e-1 -b 8388608 -sr 0.05
python main.py train -p ../data/toy/toy -d toy -de 0 1 2 3 -rk 16 -hs 17 -sp results/toy_r16_h17_sr0.05 -e 500 -lr 1e-1 -b 8388608 -sr 0.05
python main.py train -p ../data/toy/toy -d toy -de 0 1 2 3 -rk 16 -hs 17 -sp results/toy_r16_h17_sr0.1 -e 500 -lr 1e-1 -b 8388608 -sr 0.1
python main.py train -p ../data/toy/toy -d toy -de 0 1 2 3 -rk 16 -hs 17 -sp results/toy_r16_h17_sr0.2 -e 500 -lr 1e-1 -b 8388608 -sr 0.2
