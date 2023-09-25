from OCT_train_new_dataset import trainModels
# torch.manual_seed(0)
# ===================================================================================

if __name__ == '__main__':
    trainModels(model='SOASNet',
                input_dim=1,
                epochs=50,
                data_directory='/home/ubuntu',
                data_set='dataset',
                # data_directory='C:\\Users\\sbng352\\Desktop',
                # data_set='OCT_CNNtraining',
                width=16,
                depth=4,
                depth_limit=6,
                repeat=3,
                l_r=1e-3,
                l_r_s=True,
                train_batch=8,
                shuffle=True,
                data_augmentation_train='all',
                data_augmentation_test='all',
                loss='dice',
                norm='bn',
                log='Relayer_Settings',
                class_no=12,
                )
