# Section 6: Convolutional Neural Network

## Create data loader-

* Implementation - [nbviewer](http://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/06.01_CNN_create_data_loader.ipynb?flush_cache=true)

    ??? info "Alternative link"
        Alternative link - [source repository](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/06.01_CNN_create_data_loader.ipynb)


* Data - [Image_data](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/Data/img_data)


## Define CNN model architecture -

* Implementation - [nbviewer](http://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/06.02_CNN_architecture.ipynb?flush_cache=true)

    ??? info "Alternative link"
        Alternative link - [source repository](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/06.02_CNN_architecture.ipynb)


    !!! note "Update: add relu in forward method"

        ```python hl_lines="6-9"
        def forward(self, x):
            x = self.conv_pool_01(x)
            x = self.conv_pool_02(x)
            x = self.Flatten(x)
            x = self.FC_01(x)
            x = F.relu(x)
            x = self.FC_02(x)
            x = F.relu(x)    
            x = self.FC_03(x)
            return x
        ```


* Data - [Image_data](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/Data/img_data)


 


## Train CNN model -


* Implementation - [nbviewer](http://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/06.03_train_CNN.ipynb?flush_cache=true)

    ??? info "Alternative link"
        Alternative link - [source repository](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/06.03_train_CNN.ipynb)

* Data - [Image_data](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/Data/img_data)




## Evaluate CNN model -


* Implementation - [nbviewer](http://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/06.04_evaluate_CNN.ipynb?flush_cache=true)

    ??? info "Alternative link"
        Alternative link - [source repository](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/06.04_evaluate_CNN.ipynb)

* Data - [Image_data](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/Data/img_data)




## Predict using CNN model - 


* Implementation - [nbviewer](http://nbviewer.org/github/c17hawke/Pytorch-basics/blob/main/codebase/06.05_predict_using_CNN.ipynb?flush_cache=true)

    ??? info "Alternative link"
        Alternative link - [source repository](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/06.05_predict_using_CNN.ipynb)

* Data - [Image_data](https://github.com/c17hawke/Pytorch-basics/blob/main/codebase/Data/img_data)


