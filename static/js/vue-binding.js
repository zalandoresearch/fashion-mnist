ZMNIST_RESULT = 'zalando-mnist-76.json';
ORIGINAL_MNIST_RESULT = 'zalando-mnist-52.json';

function parseResult(raw_str) {
    var lstBM = JSON.parse('[' + raw_str.trim().split('\n').join(',') + ']');
    $.each(lstBM, function (index) {
        $.each(Object.keys(lstBM[index]), function (idx, key) {
            lstBM[index]['_' + key] = lstBM[index][key]
        });
        lstBM[index]['_time_per_repeat'] = (lstBM[index]['done_time'] - lstBM[index]['start_time']) / lstBM[index]['num_repeat'];
        lstBM[index]['time_per_repeat'] = moment.duration(Math.ceil(lstBM[index]['_time_per_repeat']), "seconds").format("h:mm:ss", {trim: false});
        lstBM[index]['parameter'] = JSON.stringify(lstBM[index]['parameter']);
        lstBM[index]['mean_accuracy'] = lstBM[index]['mean_accuracy'].toFixed(3);
        lstBM[index]['std_accuracy'] = lstBM[index]['std_accuracy'].toFixed(3);
        lstBM[index]['start_time'] = moment(lstBM[index]['start_time'] * 1000).fromNow();
        lstBM[index]['done_time'] = moment(lstBM[index]['done_time'] * 1000).fromNow();
    });
    return lstBM
}

function loadResult(fn, cb) {
    $.ajax({
        type: "GET",
        url: fn,
        dataType: "text",
        success: function (data) {
            cb(parseResult(data));
        }
    });
}

const vm = new Vue({
    el: '#query2sku-ui',
    data: {
        bm_data: {
            'zmnist': [],
            'mnist': [],
            'merge': []
        },
        sortKey: 'done_time',
        curDataName: 'zmnist',
        search: '',
        sortOrder: -1,
        datasets: {
            'zmnist': 'Fashion MNIST',
            'mnist': 'Original MNIST',
            'merge': 'Side-by-Side'
        },
        col_name_desc: {
            'name': 'Name',
            'parameter': 'Parameter',
            'mean_accuracy': 'Accuracy (mean)',
            'std_accuracy': 'Accuracy (std)',
            'time_per_repeat': 'Training time',
            'num_repeat': 'Repeats',
            'score': 'Score per repeat',
            'start_time': 'Job start',
            'done_time': 'Job Done',
            'm_mean_accuracy': 'MNIST Accuracy (mean)',
            'm_std_accuracy': 'MNIST Accuracy (std)',
            'z_mean_accuracy': 'Fashion Accuracy (mean)',
            'z_std_accuracy': 'Fashion Accuracy (std)'
        },
        col_show_name: {
            'zmnist': ['name', 'parameter', 'mean_accuracy', 'std_accuracy',
                'time_per_repeat', 'num_repeat', 'start_time', 'done_time'],
            'mnist': ['name', 'parameter', 'mean_accuracy', 'std_accuracy',
                'time_per_repeat', 'num_repeat', 'start_time', 'done_time'],
            'merge': ['name', 'parameter', 'z_mean_accuracy', 'm_mean_accuracy', 'z_std_accuracy', 'm_std_accuracy']
        }
    },
    ready: function () {
        loadResult(ZMNIST_RESULT, function (data) {
            vm.bm_data['zmnist'] = data;
            loadResult(ORIGINAL_MNIST_RESULT, function (data) {
                vm.bm_data['mnist'] = data;
                vm.bm_data['merge'] = vm.merge();
            });
        });
    },
    methods: {
        sortBy: function (sortKey) {
            this.sortOrder *= -1;
            this.sortKey = sortKey;
        },
        merge: function () {
            var tmp = {};
            var result = [];
            $.each(this.bm_data['mnist'], function (idx, data) {
                tmp[data['name'] + data['parameter']] = {
                    'm_mean_accuracy': data['mean_accuracy'],
                    'm_std_accuracy': data['std_accuracy'],
                    'z_mean_accuracy': 0,
                    'z_std_accuracy': 0,
                    'name': data['name'],
                    'parameter': data['parameter']
                };
            });
            $.each(this.bm_data['zmnist'], function (idx, data) {
                if (data['name'] + data['parameter'] in tmp) {
                    tmp[data['name'] + data['parameter']]['z_mean_accuracy'] = data['mean_accuracy'];
                    tmp[data['name'] + data['parameter']]['z_std_accuracy'] = data['std_accuracy'];
                    tmp[data['name']] = data['name'];
                    result.push(tmp[data['name'] + data['parameter']])

                }
            });

            $.each(result, function (index) {
                $.each(Object.keys(result[index]), function (idx, key) {
                    result[index]['_' + key] = result[index][key]
                });
            });

            return result;
        }
    }
});