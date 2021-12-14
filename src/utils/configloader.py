import yaml

def load_config(file_path):
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    configpath = '../../train_config.yml'
    data = load_config(configpath)
    print(data['common_config']['seed'])
    #print(data.learning_rate)