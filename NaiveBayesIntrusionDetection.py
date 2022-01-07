from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def y_assignment(y_base):
    new_y_base = []

    for y in y_base:
        #NORMAL assegno 0
        if y == 'normal.':
            new_y = 'normal.'
            new_y_base.append(new_y)
        #DOS assegno 1
        elif y == 'back.' or y == 'land.' or y == 'neptune.' or y == 'pod.' or y == 'smurf.' or y == 'teardrop.' or y =='apache2.' or y =='mailbomb.' or y =='processtable.' or y =='udpstorm.':
            new_y = 'dos.'
            new_y_base.append(new_y)
        #R2L assegno 2
        elif y == 'ftp_write.' or y == 'guess_passwd.' or y == 'imap.' or y == 'multihop.' or y == 'phf.' or y == 'spy.' or y == 'warezclient.' or y == 'warezmaster.' or y =='named.' or y =='sendmail.' or y == 'snmpgetattack.' or y == 'snmpguess.' or y == 'worm.' or y == 'xlock.' or y == 'xsnoop.':
            new_y = 'r2l.'
            new_y_base.append(new_y)
        #U2R assegno 3
        elif y == 'buffer_overflow.' or y == 'loadmodule.' or y == 'perl.' or y == 'rootkit.' or y == 'httptunnel.' or y == 'sqlattack.' or y == 'xterm.' or y == 'ps.':
            new_y = 'u2r.'
            new_y_base.append(new_y)
        #Probing assegno 4
        elif y == 'ipsweep.' or y == 'mscan.' or y == 'nmap.' or y == 'portsweep.' or y == 'saint.' or y == 'satan.':
            new_y = 'probe.'
            new_y_base.append(new_y)

    return np.array(new_y_base)


def before_classification(x_train, x_test, y_b_train, y_b_test):
    classifier = GaussianNB()
    c = classifier.fit(x_train, y_b_train)

    y_train_pred = c.predict(x_train)
    y_test_pred = c.predict(x_test)

    train_accuracy, test_accuracy = accuracy_confusion_matrix(y_b_train, y_train_pred, y_b_test, y_test_pred)

    train_accuracy = round(train_accuracy*100, 2)
    test_accuracy = round(test_accuracy*100, 2)

    return train_accuracy, test_accuracy


def after_classification(x_train, x_test, y_train, y_test):
    y_train = y_train.tolist()


    classifier = GaussianNB()
    c = classifier.fit(x_train, y_train)

    y_train_pred = c.predict(x_train)
    y_a_train_pred = y_assignment(y_train_pred)
    y_a_train = y_assignment(y_train)

    y_test_pred = c.predict(x_test)
    y_a_test_pred = y_assignment(y_test_pred)
    y_a_test = y_assignment(y_test)

    train_accuracy, test_accuracy = accuracy_confusion_matrix(y_a_train, y_a_train_pred, y_a_test, y_a_test_pred)

    train_accuracy = round(train_accuracy*100, 2)
    test_accuracy = round(test_accuracy*100, 2)

    return train_accuracy, test_accuracy

def accuracy_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred):
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    #confusion matrix
    labels = ['normal.', 'dos.', 'r2l', 'u2r.', 'probe.']
    cm = confusion_matrix(y_test, y_test_pred, normalize='true', labels=['normal.', 'dos.', 'r2l.', 'u2r.', 'probe.'])

    dataFrame = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(dataFrame, annot=True, fmt='.2%')
    print(cm)

    return train_accuracy, test_accuracy

def function(name):
    if name == '1':
        before = before_classification(x_train, x_test, y_b_train, y_b_test)
        print("Before -> Train: ", before[0], "\nTest: ", before[1])
    elif name =='2':
        after = after_classification(x_train, x_test, y_train, y_test)
        print("After -> Train: ", after[0], "\nTest: ", after[1])


if __name__ == '__main__':

    train = pd.read_csv("kddcup.data_10_percent_corrected", header=None,
                   names = ['duration', 'protocol_type', 'service','flag', 'src_bytes', 'dst_bytes',
                             'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                             'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                             'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                             'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                             'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                             'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                             'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                             'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Class'])

    test = pd.read_csv("corrected", header=None,
                    names = ['duration', 'protocol_type', 'service','flag', 'src_bytes', 'dst_bytes',
                             'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                             'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                             'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                             'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                             'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                             'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                             'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                             'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Class'])

    dataSet_total = pd.concat([train,test], ignore_index=True)

    dataSet_total = pd.DataFrame(dataSet_total, columns=['duration', 'protocol_type', 'service','flag', 'src_bytes', 'dst_bytes',
                             'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                             'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                             'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                             'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                             'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                             'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                             'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                             'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Class'])

    x_dataSet = dataSet_total.drop(columns=['Class'])
    y_dataSet = dataSet_total['Class']


    le = LabelEncoder()
    x_dataSet['protocol_type'] = le.fit_transform(x_dataSet['protocol_type'])
    x_dataSet['service'] = le.fit_transform(x_dataSet['service'])
    x_dataSet['flag'] = le.fit_transform(x_dataSet['flag'])

    continuous = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    discrete = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']


    disc = KBinsDiscretizer(n_bins=24, encode='onehot-dense')
    app = disc.fit_transform(x_dataSet[continuous])
    app = pd.DataFrame(app)
    x_dataSet = pd.concat([x_dataSet[discrete], app], axis=1)


    x = pd.DataFrame(x_dataSet)
    y = pd.DataFrame(y_dataSet)

    x_train = x.iloc[:494021,:].values
    x_test = x.iloc[494021:,:].values

    y_train = y.iloc[:494021,:].values.ravel()
    y_test = y.iloc[494021:,:].values.ravel()



    y_b_train = y_assignment(y_train)
    y_b_test = y_assignment(y_test)


    experiment = input('Inserisci 1 per before o 2 per after \n')
    function(experiment)


    plt.show()
