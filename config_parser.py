#!usr/local/python3
'''
    write the config file for notMNIST project programmatically
'''

from configparser import ConfigParser

filename = 'config.ini'
config = ConfigParser()

def write_config():
    config['Properties'] = dict(image_size = 45,
                                train_size = 200000,
                                valid_size = 10000,
                                test_size  = 10000)

    with open(filename, 'w') as f:
        config.write(f)

def read_config(filename = 'config.ini'):
    config.read(filename)
    _dict = config['Properties']
    image_size = _dict['image_size']
    train_size = _dict['train_size']
    valid_size = _dict['valid_size']
    test_size  = _dict['test_size']
    return image_size, train_size, valid_size, test_size

def main():
    write_config()
    print('Done writing the config file')
    image_size, train_size, valid_size, test_size = read_config()
    print('Done reading')
    print(image_size, train_size, valid_size, test_size)

if __name__ == '__main__': main()