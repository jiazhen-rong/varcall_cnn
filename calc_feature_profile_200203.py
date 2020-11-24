import numpy as np
import pandas as pd
import pysam
import pysamstats
import io
import os
import gzip
from datetime import datetime
import time
import re
import matplotlib.pyplot as plt
import cProfile
import pstats
import sys

'''
Function: Read gzipped VCF files.
args: path - path to the VCF file
'''
def read_vcf(path):
    with gzip.open(path, 'r') as f: 
        lines = [l.decode('UTF-8') for l in f if not l.startswith(b'##')]  # ignore the begining lines 
    #return lines
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

# for gzipped files
def read_bed(path):
    with gzip.open(path, 'r') as f: 
        lines = [l.decode('UTF-8') for l in f if not l.startswith(b'##')]  # ignore the begining lines 
    #return lines
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'CHROM': str, 'START': int, 'END': int},
        sep='\t', header=None,names=['CHROM','START','END']
    )

'''
Function: Assign label to true variants and false variants.
          Filtered out Indels, only looking at SNPs for now.
args: NA
'''
@profile
def assign_label():
    # convert vcf.gz to dataframe
    gatk_in_path = "../gatk-hc.chr20.vcf.gz"
    ture_in_path = "../msb2.b37m.chr20.vcf.gz"
    confi_region_path = "../msb2.b37m.chr20.bed.gz"
    gatk_df = read_vcf(gatk_in_path)
    true_df = read_vcf(ture_in_path)
    
    # 1. assign label
    true_label = np.intersect1d(gatk_df['POS'],true_df['POS']) # in both
    gatk_df['label'] = 0 # false GATK
    gatk_df.loc[gatk_df['POS'].isin(true_label),'label'] = 1 # true variant
    
    # 2. Filter out INDELs
    gatk_df = gatk_df.loc[(gatk_df['REF'].apply(len) == 1) & (gatk_df['ALT'].apply(len) == 1)]
    
    # 3. filter for locations that are only in confident region
    confi_region = read_bed(confi_region_path)
    start_array  = np.array(confi_region['START'])
    end_array = np.array(confi_region['END'])
    keep_index = np.array([]) # initialize empty array
    
    for start, end in zip(start_array,end_array):
        # add on location of the pos in confident region
        keep_index = np.append(keep_index,np.where(gatk_df['POS'].between(start,end) == True)[0])
    # keep unique elements
    keep_index = np.unique(keep_index)
    
    gatk_df = gatk_df.iloc[keep_index,:]

    return gatk_df

'''
To create balanced dataset. 
Randomly downsampling true variants 
so that # true var = # false var
'''
@profile
def downsample(gatk_df,pos,DEBUG=False):
    if DEBUG:  # selected region of variants
        temp_df = gatk_df.loc[gatk_df['POS'].isin(test_pos)]
        fp_num = sum(temp_df['label'] == 0) # get all fp samples
        bal_tp_pos = temp_df.loc[temp_df['label'] == 1]['POS'].sample(n=fp_num, random_state=1).tolist() # randomly get equal amount of tp samples
        fp_pos = temp_df.loc[temp_df['label'] == 0]['POS'].tolist()
    else: # all variants
        fp_num = sum(gatk_df['label'] == 0) # get all fp samples
        bal_tp_pos = gatk_df.loc[gatk_df['label'] == 1]['POS'].sample(n=fp_num, random_state=1).tolist() 
        fp_pos = gatk_df.loc[gatk_df['label'] == 0]['POS'].tolist() 
    
    return bal_tp_pos + fp_pos


'''
Calculate features 
'''
@profile
def calc_features(gatk_df, samfile, fastafile, window_size = 10, chrom = '20', DEBUG = False, test_pos = []):
    # for DEBUG
    if(DEBUG): # first two variants
        gatk_pos = downsample(gatk_df,test_pos,DEBUG)
        #label = gatk_df.loc[gatk_df['POS'].isin(gatk_pos),'label']
        #print(label)
    else: # all gatk variants
        gatk_pos = downsample(gatk_df,gatk_df['POS'].tolist(),DEBUG)
        label = gatk_df.loc[gatk_df['POS'].isin(gatk_pos),'label']
        #label = gatk_df['POS'].tolist()
        
    feature_df = pd.DataFrame(columns=['variant_POS','alt_ratio','norm_read_depth','quality_ratio',
                                       'dir_ratio','base_qual',
                                       'BaseQRankSum','ClippingRankSum','MQRankSum','label']) 
    # calc average reads depth across the whole chrom ~ take about 4.5 min
    if DEBUG:
        avg_depth = 46  ### for speed up & test only
    else:
        read_stats = pysamstats.load_coverage(samfile, chrom=chrom,pad=True)
        avg_depth = sum(read_stats['reads_all'])/len(read_stats) 
    
    for each_pos in gatk_pos:
        widnow_start = each_pos - int(window_size/2) 
        widnow_end = each_pos + int(window_size/2)
        label = gatk_df.loc[gatk_df['POS'] == each_pos,'label'].values[0]
        
        # Add in hand-crafted features from GATK
        info = re.split(r';+|=',gatk_df.loc[gatk_df['POS']==each_pos,'INFO'].tolist()[0])
        if 'BaseQRankSum' in info:
            bq_sum = info[info.index('BaseQRankSum')+1]
        else:
            bq_sum = 0
        if 'ClippingRankSum' in info:
            cr_sum = info[info.index('ClippingRankSum')+1]
        else:
            cr_sum = 0
        if 'MQRankSum' in info:
            mqr_sum = info[info.index('MQRankSum')+1]
        else:
            mqr_sum = 0
            
        alt_list = []
        norm_depth_list = []
        qual_list = []
        dir_list = []
        base_qual_list = []
        
        # get reads info via pysam at each var window
        for pileupcolumn in samfile.pileup(chrom, widnow_start, widnow_end, truncate=True, fastafile=None, compute_baq=False, stepper="samtools"):
            # reference base
            ref_base = fastafile.fetch(chrom,pileupcolumn.reference_pos, pileupcolumn.reference_pos+1) 
            map_nt_num = pileupcolumn.get_num_aligned() # number of reads that aligned to the current base position 
            norm_depth = map_nt_num/avg_depth # normalized read depth at current base position 
            norm_depth_list.append(norm_depth)
            
            seq = pileupcolumn.get_query_sequences()
            seq_upper = np.array([elem.upper() for elem in pileupcolumn.get_query_sequences()])
            
            quality = np.array(pileupcolumn.get_query_qualities()) # quality of each read here
            ### newly added, encode mutation prob
            prob_qual = 1-10**(np.array(pileupcolumn.get_query_qualities())/-10)
            # in case of empty sequence 
            if len(prob_qual) == 0:
                prob_qual = 0
            else:
                prob_qual = prob_qual[seq_upper != ref_base] # extract mutation 
                # in case of no mutation variant
                if len(prob_qual) == 0:
                    prob_qual = 0
                else:
                    prob_qual = np.mean(prob_qual)
                    
            base_qual_list.append(prob_qual) # newly added feature - probability
            
            qual_ratio = sum(quality[seq_upper != ref_base])/sum(quality) # weighted quality ratio
            qual_list.append(qual_ratio)
            # direction ratio - forward/reverse
            try:
                dir_ratio = sum([nt[0].isupper() if len(nt) > 0 else 0 for nt in seq])/ \
            (sum([nt[0].islower() if len(nt) > 0 else 0 for nt in seq])+ \
             sum([nt[0].isupper() if len(nt) > 0 else 0 for nt in seq]))
                dir_list.append(dir_ratio)
            except Exception as err:
                print("Error:",err)
                print("start",widnow_start)
                print("seq",seq)
                dir_list.append(0)
                
            alt_num = 0 
            
            for pileupread in pileupcolumn.pileups: # check each read for alteratio and other features
                if pileupread.is_del or pileupread.is_refskip: # deletion/insertion? base *
                    alt_num += 1
                    #print('IS DEL',pileupcolumn.reference_pos+1)
                
                if not pileupread.is_del and not pileupread.is_refskip: # SNP case
                    current_nt=pileupread.alignment.query_sequence[pileupread.query_position]
                    if current_nt != ref_base:
                        alt_num += 1
                    
                #if pileupread.is_refskip: # insertion?
                    #print('IS_REFSKIP',pileupcolumn.reference_pos+1)
                    #alt_num += 1
                    
                #if pileupread.indel !=0: # deletion or insertion indel
                    #alt_num += 1
                    
            if map_nt_num == 0:
                alt_ratio = 1 # deletion at the bp location
                
            else:
                alt_ratio = alt_num/map_nt_num
            alt_list.append(alt_ratio)


        feature_df = feature_df.append({'variant_POS':each_pos,'alt_ratio':alt_list,
                                        'norm_read_depth':norm_depth_list,'quality_ratio':qual_list,
                                        'dir_ratio':dir_list,'base_qual':base_qual_list,
                                        'BaseQRankSum':bq_sum,
                                        'ClippingRankSum':cr_sum,'MQRankSum':mqr_sum,
                                        'label':label},
                                       ignore_index=True)
    return feature_df


def save_result(path,feature_df):
    feature_df.to_csv(path+".csv",sep='\t')
    train_df=feature_df.sample(frac=0.8,random_state=42) # 80% data
    temp_df=feature_df.drop(train_df.index)
    val_df=temp_df.sample(frac=0.5,random_state=42) # 10% data
    test_df=temp_df.drop(val_df.index) # 10% data
    train_path = path+"_train"+".csv"
    val_path = path+"_val"+".csv"
    test_path = path+"_test"+".csv"
    train_df.to_csv(train_path,sep='\t')
    val_df.to_csv(val_path,sep='\t')
    test_df.to_csv(test_path,sep='\t')
    return print("Spliting dataset is done")

if __name__ == "__main__":
    
    output_file = sys.argv[1]
    start = datetime.now()
    gatk_df = assign_label()
    end = datetime.now()
    total_time = end-start
    print('Time elapsed (hh:mm:ss.ms) {}'.format(total_time))    
    
#    samfile = pysam.AlignmentFile('../msb2.b37m.chr20.bam','rb') 
#    fastafile = pysam.FastaFile('../chr20.fa')
#    start = datetime.now()
#    feature_df = calc_features(gatk_df, samfile, fastafile, window_size = 33, DEBUG = False, test_pos = [])
#    end = datetime.now()
#    total_time = end-start
#    print('Time elapsed (hh:mm:ss.ms) {}'.format(total_time))
#    samfile.close()
#    fastafile.close()
    
#    save_result(output_file,feature_df)

