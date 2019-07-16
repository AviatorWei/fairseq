#!/bin/bash
#To set support set directory
SUP_ROOT=sup_set_de2en
#Head & Tail of one-the-fly dataset directory
HEAD=de2en_
TAIL=
#Total number of test-set
TOT=6750

#Path to dictionary
DICT=/home/shawn/Downloads/fairseq-GenSup/data-raw/iwslt14.tokenized.de-en

#Path to checkpoint
CP=/home/shawn/Downloads/fairseq-master/checkpoints/fconv/checkpoint_best.pt
#ARCHITECTURE, identical to training data
ARCH=fconv_iwslt_de_en

#Model Save Path, useful for generation
SAVE=checkpoints/fconv-ft

# For support-set generation, be sure to set a K for KNN inside generate_support_set.py
# And modify need_embedded & need_tested manually (to be done by passing args)
######################################################################################################
# Caution: This step is based on modification on /fairseq/sequence_generator.py by adding encoder_outputs to return var.
echo Start Generating Support-sets
#python3 generate_support_set.py data-raw/iwslt14.tokenized.de-en/ --path checkpoints/fconv/checkpoint_best.pt --beam 5 --remove-bpe --quiet --dataset-impl raw

# We'll have to put dictionaries into each support-set folder
echo Inserting Dictionary...
for ((i = 0; i < $TOT; i++))
do
	echo ${HEAD}${i}${TAIL}
	#cp $DICT/dict.de.txt ./$SUP_ROOT/${HEAD}${i}${TAIL}
	#cp $DICT/dict.en.txt ./$SUP_ROOT/${HEAD}${i}${TAIL}
done
 
echo Done inserting dictionaries...
echo

for ((i = 0; i < $TOT; i++))
do
	echo Starting On-the-fly training...
	echo On Sample:	${HEAD}${i}${TAIL}
	# Training on-the-fly, be aware of the learning rate & max tokens (max tokens should be less than given support set?)
	# We should also consider the optimizer 
	#python3 train.py $SUP_ROOT/${HEAD}${i}${TAIL} --restore-file $CP -a $ARCH --max-tokens 40 --reset-optimizer --save-dir $SAVE/${HEAD}${i}${TAIL} --raw-text -
done

# Here's the problem: How can we concatenate the scoring step & generating step by aligning sentences?
# A: We can generate the test-set together while generating support-set 
# (write a new file of all test-sentences with new order according to SUP_SET indices)

#python3 train.py sup_set_de2en/de2en_18 --restore-file /home/shawn/Downloads/fairseq-master/checkpoints/fconv/checkpoint_best.pt -a fconv_iwslt_de_en --max-tokens 4000 --reset-optimizer --save-dir checkpoints/fconv-fttest --raw-text
