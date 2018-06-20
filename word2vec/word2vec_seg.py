import jieba

file_train_read = []
with open('/Users/chuang/code/statistical-learning/word2vec/corpus') as file_train_row:
    for line in file_train_row:
        file_train_read.append(line)

# 分词
file_train_seg = []
# file_train_seg.append('test')
for i in range (len(file_train_read)) :
    file_train_seg.append([' '.join(list(jieba.cut(file_train_read[i][9:-11], cut_all = False)))])
    if i % 500 == 0 :
        print(i)

# 输出jieba分词结果
file_seg_word_done_path = 'corpus_seg_done.txt'
with open (file_seg_word_done_path, 'wb') as seg_result:
    for i in range(len(file_train_seg)):
        seg_result.write(file_train_seg[i][0].encode('utf-8'))
        seg_result.write(b'\n')

def print_list_chinese(list):
    for i in range(len(list)): 
        print(list[i])

print_list_chinese(file_train_seg[10])


