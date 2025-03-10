import os, json, re


class dataset_walker(object):

    def __init__(self, dataset, labels=False, dataroot=None, config_folder=None):
        if "[" in dataset:
            self.datasets = json.loads(dataset)
        elif type(dataset) == type([]):
            self.datasets = dataset
        else:
            self.datasets = [dataset]
            self.dataset = dataset
        self.install_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_session_lists = [config_folder + '/' + dataset + '.flist' for dataset in self.datasets]
        # self.dataset_session_lists = [os.path.join(self.install_root, 'config', dataset + '.flist') for dataset in
        #                               self.datasets]

        self.labels = labels
        if (dataroot == None):
            install_parent = os.path.dirname(self.install_root)
            self.dataroot = os.path.join(install_parent, 'data')
        else:
            self.dataroot = os.path.join(os.path.abspath(dataroot))

        # load dataset (list of calls)
        self.session_list = []
        for dataset_session_list in self.dataset_session_lists:
            f = open(dataset_session_list)
            for line in f:
                line = line.strip()
                # line = re.sub('/',r'\\',line)
                # line = re.sub(r'\\+$','',line)
                if (line in self.session_list):
                    raise RuntimeError('Call appears twice: %s' % (line))
                self.session_list.append(line)
            f.close()

    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot, *session_id_list)
            applog_filename = os.path.join(session_dirname, 'log.json')
            if (self.labels):
                labels_filename = os.path.join(session_dirname, 'label.json')
                if (not os.path.exists(labels_filename)):
                    raise RuntimeError('Cant score : cant open labels file %s' % (labels_filename))
            else:
                labels_filename = None
            call = Call(applog_filename, labels_filename)
            call.dirname = session_dirname
            yield call

    def __len__(self, ):
        return len(self.session_list)


class Call(object):
    
    def __init__(self, applog_filename, labels_filename):
        self.applog_filename = applog_filename
        self.labels_filename = labels_filename
        f = open(applog_filename)
        self.log = json.load(f)
        f.close()
        if (labels_filename != None):
            f = open(labels_filename)
            self.labels = json.load(f)
            f.close()
        else:
            self.labels = None

    def __iter__(self):
        if (self.labels_filename != None):
            for (log, labels) in zip(self.log['turns'], self.labels['turns']):
                yield (log, labels)
        else:
            for log in self.log['turns']:
                yield (log, None)

    def __len__(self, ):
        return len(self.log['turns'])


if __name__ == '__main__':
    import misc

    dataset = dataset_walker("HDCCN", dataroot="data", labels=True)
    for call in dataset:
        if call.log["session-id"] == "voip-f32f2cfdae-130328_192703":
            for turn, label in call:
                print(misc.S(turn))
