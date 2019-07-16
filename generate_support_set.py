#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from functools import reduce
import numpy as np


class EmbedStorer:
    def __init__(self):
        self.sent_str = []
        self.sent_token = []
        self.sent_len = []
        self.sent_emb = []
        self.tgt_str = []

    def add_sentence(self, str, pad_token, tgt, embed):
        #assert len(embed) == 768
        self.sent_token.append(pad_token)
        self.sent_str.append(str)
        self.sent_len.append(len(pad_token))
        self.sent_emb.append(embed)
        self.tgt_str.append(tgt)


def main(args, mem):

    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos, embed = task.inference_step(generator, models, sample, prefix_tokens)
            """
            print("****Test Hypo & Embed****")
            print(len(hypos))
            print(hypos[0])
            #print(hypos[0].keys())
            batch = embed[0]
            print(batch.keys())
            for k in batch.keys():
                print(k)
                print(len(batch[k]))
                # batch[k] is a tuple with two elements, where the
                # first element is the last encoder layer's output and the
                # second element is the same quantity summed with the input
                # embedding (used for attention)
                # The shape of both tensors is `(batch, src_len, embed_dim)`.
                print(list(batch[k][0].size()))

            embedding = batch['encoder_out'][0]
            print(len(sample['id'].tolist()))
            assert list(embedding.size())[0] == len(sample['id'].tolist())
            print("********END  Hypo********")
            """
            batch = embed[0]
            embedding = batch['encoder_out'][0]
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                # Here I memorize the tensors into memory -- WZC
                mem.add_sentence(src_str, src_tokens, target_str,
                                 reduce(lambda x,y: x + y, embedding[i].cpu().numpy()))
                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    return scorer

import faiss
import os

def calculate_KNN(base_memory, test_memory):
    import copy
    print(len(base_memory.sent_len), len(test_memory.sent_len))
    assert (len(base_memory.sent_len) == len(base_memory.sent_emb))
    assert (len(base_memory.sent_len) == len(base_memory.sent_str))
    assert (len(test_memory.sent_len) == len(test_memory.sent_emb))
    assert (len(test_memory.sent_len) == len(test_memory.sent_str))
    print("********Start  FAISS initialization*******")
    ngpus = 1#faiss.get_num_gpus()

    print("number of GPUs:", ngpus)
    dimension = base_memory.sent_emb[0].shape[0]
    print("Check dim before CPU init:", dimension)
    cpu_index = faiss.IndexFlatL2(dimension)
    print("End of CPU init, Start transformation")
    ##############
    res = faiss.StandardGpuResources()
    #gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    #    cpu_index
    #)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    print(type(base_memory.sent_emb))
    pure_emb = np.array(base_memory.sent_emb)
    print(pure_emb.shape)
    gpu_index.add(pure_emb)  # add vectors to the index
    print(gpu_index.ntotal)
    print("********End of FAISS initialization*******")
    src_sent = copy.deepcopy(base_memory.sent_str)
    tgt_sent = copy.deepcopy(base_memory.tgt_str)

    ###############################
    # NEED CONFIGURATION
    src_lang = "de"
    tgt_lang = "en"
    num_nearest_neighbor = 100  # we want to see 4 nearest neighbors
    fp = open("de2en_log.txt", 'w+')
    #################################

    split = ["test", "train", "valid"]
    tail = "." + src_lang + "-" + tgt_lang + "."
    k = num_nearest_neighbor
    print("********Start FAISS calculate of support set: k = {}*******".format(k))
    os.mkdir("sup_set_de2en")
    len_test = len(test_memory.sent_len)
    for idx in range(len_test):
        str_q = test_memory.sent_str[idx]
        tgt_q = test_memory.tgt_str[idx]
        print("******Sentence {}******".format(idx), file=fp)
        print("Src sentence:", str_q, file=fp)
        print("Golden Answer:", tgt_q, file=fp)
        print("Src length", base_memory.sent_len[idx], file=fp)

        folder = "sup_set_de2en/de2en_{}".format(idx)
        os.mkdir(folder)
        sup_set = open(folder+"/train"+tail+src_lang, "a+")
        val_set = open(folder+"/valid"+tail+src_lang, "a+")
        sup_gold = open(folder+"/train"+tail+tgt_lang, "a+")
        val_gold = open(folder+"/valid"+tail+tgt_lang, "a+")
        test_set = open(folder+"/test"+tail+src_lang, "a+")
        test_gold = open(folder+"/test"+tail+tgt_lang, "a+")

        xq = np.ones(shape=(1, dimension))
        xq[0] = np.array(test_memory.sent_emb[idx])
        xq = xq.astype(np.float32)
        D, I = gpu_index.search(xq, k)  # actual search
        print(I, file=fp)

        _train = [src_sent[i] for i in I[0]]
        _gold = [tgt_sent[i] for i in I[0]]
        print(_train, file=fp)
        print("\n".join(_train), file=sup_set)
        print("\n".join(_train), file=val_set)
        print("\n".join(_gold), file=sup_gold)
        print("\n".join(_gold), file=val_gold)

        print(str_q, file=test_set)
        print(tgt_q, file=test_gold)
        print("Finish writing ", folder, "No.", idx)

        sup_set.close()
        val_set.close()
        val_gold.close()
        sup_gold.close()
        test_set.close()
        test_gold.close()

    # return sent-represetnation from src_sent now
    # return (_train, test_data)


train_mem = EmbedStorer()
test_mem = EmbedStorer()
def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    args.data = "data-raw/need_embedded"
    print(args)
    print("******Start training set******")
    main(args, train_mem)
    print("******End  training  set******")
    print()
    print()
    args.data = "data-raw/need_tested"
    print(args)
    print("******Start testing set*******")
    main(args, test_mem)
    print("*******End  testing  set******")
    calculate_KNN(train_mem, test_mem)
    return


if __name__ == '__main__':
    cli_main()
