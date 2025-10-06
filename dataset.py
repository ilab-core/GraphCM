#import glob
import os
import json
import logging
#import math
#import numpy as np
#import torch
#import pprint
import utils
import random

class Dataset(object):
    """
    Bu modül, veri setini verimli bir şekilde yönetir. Veriyi belleğe yüklemeden,
    ihtiyaç duyulduğunda dosyadan okuyarak batch'ler oluşturur.
    """
    def __init__(self, args):
        self.logger = logging.getLogger("GraphCM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.dataset = args.dataset
        self.data_dir = os.path.join('data', self.dataset)
        self.args = args
        
        self.train_path = os.path.join(self.data_dir, 'train_per_query_quid.txt')
        self.valid_path = os.path.join(self.data_dir, 'valid_per_query_quid.txt')
        self.test_path = os.path.join(self.data_dir, 'test_per_query_quid.txt')
        
        label_path = os.path.join(self.data_dir, 'human_label_for_GraphCM_per_query_quid.txt')
        self.label_path = label_path if os.path.exists(label_path) else None

        self.trainset_size = utils.count_lines(self.train_path) if os.path.exists(self.train_path) else 0
        self.validset_size = utils.count_lines(self.valid_path) if os.path.exists(self.valid_path) else 0
        self.testset_size = utils.count_lines(self.test_path) if os.path.exists(self.test_path) else 0
        self.labelset_size = utils.count_lines(self.label_path) if self.label_path else 0
        
        self.query_qid = utils.load_dict(self.data_dir, 'query_qid.dict')
        self.url_uid = utils.load_dict(self.data_dir, 'url_uid.dict')
        self.vtype_vid = utils.load_dict(self.data_dir, 'vtype_vid.dict')
        self.query_size = len(self.query_qid)
        self.doc_size = len(self.url_uid)
        self.vtype_size = len(self.vtype_vid)
        self.padding_uid = 0
        self.logger.info('Train set size: {} sessions.'.format(self.trainset_size))
        self.logger.info('Dev set size: {} sessions.'.format(self.validset_size))
        self.logger.info('Test set size: {} sessions.'.format(self.testset_size))
        self.logger.info('Label set size: {} sessions.'.format(self.labelset_size))

    def _parse_line_to_session(self, line, mode):
        """Tek bir satırı ayrıştırıp bir session dictionary'sine dönüştürür."""
        attr = line.strip().split('\t')
        
        # 5 sütunlu formata göre okuma
        qids = [int(attr[1].strip())]
        uids = json.loads(attr[2].strip())
        vids = json.loads(attr[3].strip())
        clicks = json.loads(attr[4].strip())

        # Gerçek ilan sayısını, PADDING ID'Sİ (0) olmayanlara bakarak buluyoruz.
        actual_len = len([uid for uid in uids if uid != self.padding_uid])
        
        # Maskeyi anlık olarak üretiyoruz: Gerçek veriler için 1, dolgu için 0.
        mask = [1] * actual_len + [0] * (self.max_d_num - actual_len)

        last_rank = 0
        for idx, click in enumerate(clicks):
            last_rank = idx + 1 if click else last_rank

        return {
            'sid': attr[0].strip(),
            'qids': qids,
            'uids': uids,
            'vids': vids,
            'clicks': clicks,  
            'mask': mask,
            'last_rank': last_rank,
        }

    def _one_mini_batch(self, data):
        """
        Bir grup session verisinden modelin beklediği formatta mini-batch oluşturur.
        """
        batch_data = {'raw_data': data, 'qids': [], 'uids': [], 'vids': [], 'clicks': [], 'masks': [],
                    'last_ranks': [], 'true_clicks': []}
        for sample in data:
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['vids'].append(sample['vids'])
            batch_data['clicks'].append(sample['clicks'])
            batch_data['masks'].append(sample['mask'])
            batch_data['last_ranks'].append(sample['last_rank'])
            batch_data['true_clicks'].append(sample['clicks'])
            
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Veri setini belleğe yüklemeden, dosyadan satır satır okuyarak
        mini-batch'ler üreten (yield eden) verimli versiyon.
        """
        if set_name == 'train': path = self.train_path
        elif set_name == 'valid': path = self.valid_path
        elif set_name == 'test': path = self.test_path
        elif set_name == 'label': path = self.label_path
        else: raise NotImplementedError('Set name {} is not supported'.format(set_name))

        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            return

        with open(path, 'r') as f:
            batch_sessions = []
            for line in f:
                try:
                    if len(line.strip().split('\t')) != 5:
                        self.logger.warning(f"Beklenenden farklı sütun sayısına sahip satır atlandı: {line.strip()}")
                        continue
                    session_data = self._parse_line_to_session(line, mode=set_name)
                    batch_sessions.append(session_data)
                    if len(batch_sessions) == batch_size:
                        if shuffle: random.shuffle(batch_sessions)
                        yield self._one_mini_batch(batch_sessions)
                        batch_sessions = []
                except (IndexError, json.JSONDecodeError, ValueError):
                    self.logger.warning(f"Hatalı formatlı satır atlandı: {line.strip()}")
                    continue
            
            if len(batch_sessions) > 0:
                if shuffle: random.shuffle(batch_sessions)
                yield self._one_mini_batch(batch_sessions)

