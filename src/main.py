from sklearn import datasets

def main():
    iris = datasets.load_iris()
    print(f'✅ Loaded dataset with shape: {iris.data.shape}')

if __name__ == '__main__':
    main()
