import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

#在dataset15.csv数据集中只取前120000条会话
with open('tmall_data.csv', 'w') as tmall_data:
    with open('tmall/dataset15.csv', 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data.write(header)
        for line in tmall_file:
            data = line[:-1].split('\t')
            if int(data[2]) > 120000:
                break
            tmall_data.write(line)

print("-- Starting @ %ss" % datetime.datetime.now())

#将tmall_data.csv中数据进行统计，构造sess_clicks字典，其中每个键值对的key为sessionID,value为对应的session,sess_data字典中key存放sessionID，value存放每个会话最后的结束时间，
with open('tmall_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='\t')      #读取csv文件，以\t为分隔符分开
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['SessionId'])   #sessid存放会话ID
        if curdate and not curid == sessid:   #当curdate大于0且curid!=sessid时才会触发
            date = curdate
            sess_date[curid] = date
        curid = sessid
        item = int(data['ItemId'])
        curdate = float(data['Time'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = float(data['Time'])
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

with open('Tmall_category.csv',"r") as f:              #读出商品的item_ID对应的种类category_Id
    reader = csv.DictReader(f)
    item_category = {}
    for data in reader:
        item_id = int(data['item_id'])
        item_category[item_id] = int(data['category_id'])

# Filter out length 1 sessions
#iid_counts字典中存放的keys为itemID，values为每个item在数据集中出现的总次数
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
#iid_counts字典中存放的keys为itemID，values为每个item在数据集中出现的总次数
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

#将iid_counts字典按value值进行升序排序，并放入列表中，列表中的每个元素为一个元组，其中存放itemID，和item在所有会话中出现的次数
sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
#对sess_clicks中的每个会话，只保留item在每个会话中出现次数大于4次的item，且只保留筛选后会话长度在2~40之间的会话，并更新到sess_clicks中
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))     #计算在每个会话中出现次数大于等于5次的itemID
    if len(filseq) < 2 or len(filseq) > 40:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
#找出最后一个会话的结束时间
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# the last of 100 seconds for test      测试的最后100秒
splitdate = maxdate - 100

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
#最后100秒前的item作为训练会话，100秒后的作为测试会话
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
new_item_category = {}
# Convert training sessions to sequences and renumber items to start from 1   将训练session重新转换为序列并从1开始
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    # 统计所有session中的itemID
    for s, date in tra_sess:       #tra_sess里存放：每个session中itemID出现次数大于4次的item且session长度在2~40之间的session和他们的时间，且结束时间在100秒前的会话，且按时间排序过
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr          #item_dict的key为itemID，value的值依次相加

                new_item_category[item_ctr] = item_category[i]     #~~~~~~构造新的商品--类别ID的字典索引  将原有的dataset15中的物品的id重新编码从1开始，对应的类别为天池数据集中的类别

                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]     #训练集的每个session的会话ID
        train_dates += [date]      #训练集的每个session的结束时间
        train_seqs += [outseq]     #里面存放每个session的itemID的转换顺序
    print('item_ctr:')
    print(item_ctr)     # 43098, 37484

    change = {}                      #将商品的类别字典中的类别ID重新编号（从"item的编号尾部 "开始），用于之后的类别信息的embedding嵌入
  #  category_ctr = 1
    category_ctr = 1 + 40727             #!!!!!类别ID与itemID不重复
    for k,v in new_item_category.items():        #new_item_category存放的是新的itemID和其对应的类别
        if v in change:
            new_item_category[k] = change[v]
        else:
            change[v] = category_ctr
            new_item_category[k] = change[v]         #new_item_category[k]实际上存放的是value的值，即类别ID
            category_ctr += 1
    print("\n\ncategory_ctr:",category_ctr)
    return train_ids, train_dates, train_seqs, new_item_category        #返回sessionID，session_data, session的新itemID序列，新的item_category

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]        #测试集的每个会话ID
        test_dates += [date]         #测试集的每个session的结束时间
        test_seqs += [outseq]        #测试集的itemID的转换顺序
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs, item_category = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):      #传入参数为itemID转换序列和会话结束时间
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):    #id表示传入多个会话列表的会话ID，从0开始，seq表示传入的会话，date对应时间
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]                 #每个会话的待预测itemID
            out_seqs += [seq[:-i]]        #会话session 经过数据增强后的session
            out_dates += [date]           #每个会话的结束时间
            ids += [id]                   #每个sessionID
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print('train and test')
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all * 1.0/(len(tra_seqs) + len(tes_seqs)))

if not os.path.exists('tmall'):
    os.makedirs('tmall')
pickle.dump(tra, open('tmall/train.txt', 'wb'))
pickle.dump(tes, open('tmall/test.txt', 'wb'))
pickle.dump(tra_seqs, open('tmall/all_train_seq.txt', 'wb'))

pickle.dump(item_category,open('tmall/category.txt', 'wb'))

# Namespace(dataset='Tmall')
# Splitting train set and test set
# item_ctr
# 40728
# train_test
# 351268
# 25898
# avg length:  6.687663052493478
