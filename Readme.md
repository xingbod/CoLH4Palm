

# Training


python main.py  -b 256 --ds "casiam" --bits 32 --model res18 --comment caRes32 --epoch 4000

python main.py  -b 256 --ds "polyu" --bits 32 --model res18 --comment poRes32 --epoch 4000 --gpu 1

python main.py  -b 256 --ds "tjppv" --bits 32 --model res18 --comment tjRes32 --epoch 4000


python main.py  -b 32 --ds "casiam" --bits 32 --model effb5 --comment caEff32 --epoch 4000

python main.py  -b 32 --ds "polyu" --bits 32 --model effb5 --comment poEff32 --epoch 4000

python main.py  -b 32 --ds "tjppv" --bits 32 --model effb5 --comment tjEff32 --epoch 4000



python main.py  -b 32 --ds "casiam" --bits 64 --model effb5 --comment caEff64
python main.py  -b 32 --ds "casiam" --bits 128 --model effb5 --comment caEff128

python main.py  -b 512 --ds "casiam" --bits 64 --model res18 --comment caRes64
python main.py  -b 512 --ds "casiam" --bits 128 --model res18 --comment caRes128

On mobilenet:
python main.py  -b 512 --ds "tjppv" --bits 128 --model mobile --comment tjMobile128 --epoch 4000

python main.py  -b 512 --ds "polyu" --bits 128 --model mobile --comment poMobile128 --epoch 4000

python main.py  -b 512 --ds "casiam" --bits 128 --model mobile --comment caMobile128 --epoch 4000