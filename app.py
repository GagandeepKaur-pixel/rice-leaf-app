{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ee79c0d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install numpy\n",
    "#!pip install pandas\n",
    "#pip install pillow\n",
    "#pip install seaborn\n",
    "#pip install opencv-python\n",
    "#pip install keras\n",
    "#pip install tensorflow\n",
    "#pip install scikit-learn\n",
    "#pip install pydot\n",
    "#pip install pandas_ml"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a986a24b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['.ipynb_checkpoints', 'Bacterial Leaf Blight', 'BLB_350', 'CNN.ipynb', 'Decision Tree_32x32_Leaf Blast.ipynb', 'Decision Tree_LeafBlast_Results_Python', 'Decision Tree_Yellow Rust_Result_Python', 'Final code.txt', 'final.txt', 'final222.txt', 'final_gagan_mam (2)-Copy1.ipynb', 'final_gagan_mam.ipynb', 'final_gagan_mam.txt', 'Gagan.h5', 'Gagan.png', 'Healthy', 'Healthy Rice Leaf', 'Image Rotation_New (1).ipynb', 'Image Rotation_New.ipynb', 'Inception _NPK_224 model.ipynb', 'Inception_ResnetV2.png', 'Inception_ResnetV2_NEW.ipynb', 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', 'Inception_Results_Python_Bacterial Leaf Blight_224.ipynb', 'Inception_Results_Python_Sheath Blight_350.ipynb', 'Inception_Results_Python_Yellow Rust 350.ipynb', 'K', 'Leaf Blast', 'Leaf Blast_Result_Python_Sheath', 'LeafBlast', 'LeafBlast350', 'LENET.png', 'LENET_BLB_128x128.ipynb', 'LENET_BLB_256x256.ipynb', 'LENET_BLB_2X2.ipynb', 'LENET_BLB_32X32.ipynb', 'LENET_BLB_4X4.ipynb', 'LENET_BLB_64X64.ipynb', 'LENET_BLB_8X8.ipynb', 'LENET_new.ipynb', 'LENET_NPK_12x12.ipynb', 'LENET_NPK_132x132.ipynb', 'LENET_NPK_24x24.ipynb', 'LENET_NPK_264x264.ipynb', 'LENET_NPK_3x3.ipynb', 'LENET_NPK_48x48.ipynb', 'LENET_NPK_6x6.ipynb', 'LENET_NPK_96x96.ipynb', 'LENET_Sheath Blight_16x16.ipynb', 'LENET_Sheath Blight_32x32.ipynb', 'LENET_Sheath Blight_4x4.ipynb', 'LENET_Sheath Blight_64x64.ipynb', 'LENET_Sheath Blight_8x8.ipynb', 'LENET_Yellow Rust_124x124.ipynb', 'LENET_Yellow Rust_16x16.ipynb', 'LENET_Yellow Rust_32x32.ipynb', 'LENET_Yellow Rust_64x64.ipynb', 'model', 'model.h5', 'model.png', 'N', 'Naive Baise_Result_Python', 'Naive Bayes (4) - Copy.ipynb', 'Naive Bayes 128x128_Sheath blight-Copy1.ipynb', 'Naive Bayes 128x128_Sheath blight.ipynb', 'Naive Bayes 16x16_Sheath blight.ipynb', 'Naive Bayes 256x256_Sheath blight.ipynb', 'Naive Bayes 32x32_Sheath blight.ipynb', 'Naive Bayes 4x4_Sheath blight.ipynb', 'Naive Bayes 64x64_Sheath blight.ipynb', 'Naive Bayes 8x8_Sheath blight.ipynb', 'Naive Bayes_NPK 16x16.ipynb', 'NPK 16x16.ipynb', 'P', 'Random Forest_LeafBlast_Result_Python', 'Random Forest_Yellow Rust_Results_Python', 'Rice_images', 'Sheath Blast 350', 'SheathBlast', 'SVM_LeafBlast_Result_Python', 'SVM_Yellow Rust_Result Python', 'VCG-16_LeafBlast350_Result_Python', 'xception_model.png', 'Xception_New-BLB_224 new.ipynb', 'Xception_New-BLB_224.ipynb', 'Xception_New-Copy2.ipynb', 'Xception_New-Copy3.ipynb', 'Xception_New.ipynb', 'Yellow Rust', 'Yellow Rust 350', 'Yellow Rust_350_VCG 16_Result_Python', 'Yellow rust_Lenet_32x32.ipynb', 'Yellow Rust_Result_Python']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import PIL\n",
    "from PIL import Image\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "import keras\n",
    "import tensorflow\n",
    "import sklearn\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "print(os.listdir(\"C:/Users/admin/Desktop/Images\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "625d61b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "images=[]\n",
    "labels=[]\n",
    "from PIL import ImageOps\n",
    "\n",
    "def load_images_from_folder(folder, label):\n",
    "    images = []\n",
    "    labels = []\n",
    "    for filename in os.listdir(folder):\n",
    "        if filename != \"Thumbs.db\":\n",
    "            img = cv2.imread(os.path.join(folder, filename))\n",
    "            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "            new_img = cv2.resize(gray, (16, 16))\n",
    "            images.append(new_img)\n",
    "    labels = [label] * len(images)\n",
    "    return images, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "f4d5e2fd",
   "metadata": {},
   "outputs": [
    {
     "ename": "error",
     "evalue": "OpenCV(4.10.0) D:\\a\\opencv-python\\opencv-python\\opencv\\modules\\imgproc\\src\\color.cpp:196: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31merror\u001b[0m                                     Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[61], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Load images from folders and create a train set\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m blast_images, blast_labels \u001b[38;5;241m=\u001b[39m load_images_from_folder(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mC:/Users/admin/Desktop/Images/N\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m1\u001b[39m)\n\u001b[0;32m      3\u001b[0m healthy_images, healthy_labels \u001b[38;5;241m=\u001b[39m load_images_from_folder(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mC:/Users/admin/Desktop/Images/Healthy Rice Leaf\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m0\u001b[39m)\n\u001b[0;32m      4\u001b[0m images \u001b[38;5;241m=\u001b[39m blast_images \u001b[38;5;241m+\u001b[39m healthy_images\n",
      "Cell \u001b[1;32mIn[59], line 11\u001b[0m, in \u001b[0;36mload_images_from_folder\u001b[1;34m(folder, label)\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m filename \u001b[38;5;241m!=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThumbs.db\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m     10\u001b[0m     img \u001b[38;5;241m=\u001b[39m cv2\u001b[38;5;241m.\u001b[39mimread(os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(folder, filename))\n\u001b[1;32m---> 11\u001b[0m     gray \u001b[38;5;241m=\u001b[39m cv2\u001b[38;5;241m.\u001b[39mcvtColor(img, cv2\u001b[38;5;241m.\u001b[39mCOLOR_BGR2GRAY)\n\u001b[0;32m     12\u001b[0m     new_img \u001b[38;5;241m=\u001b[39m cv2\u001b[38;5;241m.\u001b[39mresize(gray, (\u001b[38;5;241m16\u001b[39m, \u001b[38;5;241m16\u001b[39m))\n\u001b[0;32m     13\u001b[0m     images\u001b[38;5;241m.\u001b[39mappend(new_img)\n",
      "\u001b[1;31merror\u001b[0m: OpenCV(4.10.0) D:\\a\\opencv-python\\opencv-python\\opencv\\modules\\imgproc\\src\\color.cpp:196: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'\n"
     ]
    }
   ],
   "source": [
    "# Load images from folders and create a train set\n",
    "blast_images, blast_labels = load_images_from_folder(\"C:/Users/admin/Desktop/Images/N\", 1)\n",
    "healthy_images, healthy_labels = load_images_from_folder(\"C:/Users/admin/Desktop/Images/Healthy Rice Leaf\", 0)\n",
    "images = blast_images + healthy_images\n",
    "labels = blast_labels + healthy_labels\n",
    "\n",
    "# Convert data to numpy array and normalize pixel values\n",
    "train = np.array(images)\n",
    "train = train.astype('float32') / 255.\n",
    "\n",
    "# Reshape the input data to 1D array\n",
    "num_samples, height, width = train.shape\n",
    "flattened_train = train.reshape(num_samples, height * width)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c90efd12",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "fc415a18",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 93.67%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "# Split data into train and test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(flattened_train, labels, test_size=0.2, random_state=42)\n",
    "\n",
    "\n",
    "\n",
    "# Train a Naive Bayes classifier\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "\n",
    "# Predict the labels of the test set\n",
    "y_pred = nb.predict(X_test)\n",
    "\n",
    "# Evaluate the performance of the classifier using accuracy and confusion matrix\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy: {:.2f}%\".format(accuracy*100))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ae090180",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nnoqvXr1yq233prtttuu+JAAANBWtPgc7IkTJ2bbbbfNc889l4cffjgPP/xw5s+fn+233z7HHnvs+pgRAADajBYfwX700Ufz4IMPZoMNNmhatsEGG+Tb3/52dtppp6LDAQBAW9PiI9hDhgzJiy++uMbyl156KZtvvnmRoQAAoK1ap8BetGhR09f555+fk046KTfeeGOee+65PPfcc7nxxhtz8sknZ/Lkyet7XgAA+ECrVKvV6rttVFNT0+xj0N+6y1vL/vr2qlWr1secHzmLFi1K9+7dM7LukLSvdGztcQDel8Zly1p7BID3raG6MjNySxYuXJj6+vq33W6dzsG+5557ig0GAAAfZusU2Hvsscf6ngMAAD4U3vMnwyxdujTz5s3LihUrmi3ffvvt3/dQAADQVrU4sF9++eUceeSRueOOO9a63jnYAAB8lLX4Mn0nn3xyXnvttTzwwAPp1KlT7rzzzlx99dXZYostcuutt66PGQEAoM1o8RHsu+++O7fcckt22mmn1NTUZODAgRk9enTq6+szadKk7LfffutjTgAAaBNafAR7yZIl6d27d5KkZ8+eefnll5Mk2223XR5++OGy0wEAQBvznj7J8cknn0yS7LDDDrn88svz5z//OZdddln69etXfEAAAGhLWnyKyMknn5wFCxYkSc4+++yMGTMm1157bTp27Jhp06aVng8AANqUdfokx3eydOnSPPHEExkwYEA22mijUnN95L31SY6j+hyT9jU+yRFo23728M9bewSA923RG43ZYMtnynyS4zvp3Llzhg0b9n53AwAAHwrrFNinnnrqOu/woosues/DAABAW7dOgT1z5sx12lmlUnlfwwAAQFu3ToF9zz33rO85AADgQ6HFl+kDAADensAGAICCBDYAABQksAEAoCCBDQAABb2nwP7hD3+Yf/zHf8zGG2+cZ599NklyySWX5JZbbik6HAAAtDUtDuypU6fm1FNPzb777pvXX389q1atSpL06NEjl1xySen5AACgTWlxYF966aW54oorcuaZZ6Zdu3ZNy4cPH55Zs2YVHQ4AANqaFgf2nDlzsuOOO66xvLa2NkuWLCkyFAAAtFUtDuxNN900jzzyyBrL77jjjmyzzTYlZgIAgDZrnT4q/a+ddtpp+cIXvpBly5alWq3m97//fX70ox9l0qRJ+cEPfrA+ZgQAgDaj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qa+vz8iRI/Poo482rX/66adz4IEHpk+fPunatWt22mmn3HXXXc1m+v73v58tttgidXV16dOnTw4++OCmddVqNd/5zncyePDgdOrUKUOHDs2NN97Y7P7Tp0/PlltumU6dOmXPPffM3Llzy75o0ALtqo1pl2pWVto1W76i0j7brliw1vt0rq5IY5IlFf+wBT6AVlSTaxamWl+TbPMOP6cWrVod0+0rzRZX/u21VLZ5JpVR85JLXl29PyjoA3kO9siRIzN06NDcdNNNmThxYrN1U6ZMya233pobbrghAwYMyPz58zN//vwkq+N3v/32S8+ePTN9+vR07949l19+efbaa6889dRT6dmzZxYvXpx999033/rWt1JXV5err746BxxwQJ588skMGDAgDz74YE466aT88Ic/zK677ppXX3019957b9Pjf+Mb38hNN92UqVOnZosttsj/+3//L4cffnh69eqVPfbYI/Pnz8+nP/3pHHfccTn++OPz4IMP5stf/vLf9fWDv/ZmTcc83qFvPrv4wcxr3zOv13TKHm/+Z4asfDHPt+uxxvYdqg058o3fZkbdllla0/HvPzDA2/nlklSOeyF5s5r0aZfq9RsnG7Zb+7avrkrl4teSf+nebHF1Yo9ku9qkR00yc1kq5/8lmd+Q6nd7r//5+cj4QAZ2kmy11Vb54x//uMbyefPmZYsttshuu+2WSqWSgQMHNq275557MmvWrLz00ktNp5RceOGFufnmm3PjjTfm2GOPzdChQzN06NCm+3zrW9/KT3/609x666058cQTM2/evHTp0iX7779/unXrloEDB2bHHXdMkixZsiQXXXRR7r777owYMSJJMnjw4PzmN7/J5Zdfnj322CNTp07N4MGDc/HFF6dSqWTIkCGZNWtWJk+e/I7Pd/ny5Vm+fHnT7UWLFr33Fw/+xoU9RuWUhXfn2pemZVUq+a8OvTKjbsts3vBys+3aVVfljNd+kZpqNf/WfY9Wmhbgbfxjp1Tv6p+82pjKtQtTOfaFVKd/LNnob3LmjcZU/uX5ZMuOqX75b86v/nyP//l+m9pUu7dLzTEvpHrmhqvfGAkFfGADu1qtplKprLF8woQJGT16dIYMGZKxY8dm//33z957750keeihh7J48eJsuOGGze7z5ptv5umnn06yOpLPPffc3H777Xn++efT0NCQN998M/Pm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+tdzyy235I477si0adOSND9FZMWKFTnxxBPTr1+/1NXVZdCgQZk0aVLT/hYuXJhjjz02vXv3Tn19fUaOHJlHH320af3TTz+dAw88MH369EnXrl2z00475a677mo20/e///1sscUWqaurS58+fXLwwQc3ratWq/nOd76TwYMHp1OnThk6dGhuvPHGZvefPn16ttxyy3Tq1Cl77rln5s6dW/ZF40NFYEMbNmvWrIwZMyaf/vSn88c//jHXX399fvOb3zQL3RUrVuS8887Lo48+mptvvjlz5szJhAkT1rq/XXfdNZdccknq6+uzYMGCLFiwIF/5ylea1n/3u9/N8OHDM3PmzJxwwgk5/vjj88QTTyRJJk6cmOuuuy7Lly9v2v7aa6/NxhtvnD333HP9vABAmzFy5MgMHTo0N9100xrrpkyZkltvvTU33HBDnnzyyVxzzTUZNGhQktXxu99+++WFF17I9OnT89BDD2XYsGHZa6+98uqrryZJFi9enH333Td33XVXZs6cmTFjxuSAAw7IvHnzkiQPPvhgTjrppPzrv/5rnnzyydx5553Zfffdmx7/G9/4Rq666qpMnTo1jz32WE455ZQcfvjh+fWvf50kmT9/fj796U9n3333zSOPPJKJEyfmjDPOWM+vGG1aFfhAOuKII6rt2rWrdunSpdlXXV1dNUn1tddeq/7Lv/xL9dhjj212v3vvvbdaU1NTffPNN9e639///vfVJNU33nijWq1Wq/fcc0/T/qrVavWqq66qdu/efY37DRw4sHr44Yc33W5sbKz27t27OnXq1Gq1Wq0uW7as2rNnz+r111/ftM0OO+xQPeecc97PywC0MUcccUT1wAMPXOu68ePHV7feeutqtVqtJqn+9Kc/rVar1eoXv/jF6siRI6uNjY1r3OdXv/pVtb6+vrps2bJmyzfbbLPq5Zdf/rZzbLPNNtVLL720Wq1Wqz/5yU+q9fX11UWLFq2x3eLFi6t1dXXV+++/v9nyo48+uvrZz362Wq1Wq1/72teqW2+9dbP5Tj/99GY/O+GvtW/Vugfe0Z577pmpU6c2W/a73/0uhx9+eJLkoYceyn/913/l2muvbVpfrVbT2NiYOXPmZOutt87MmTNzzjnn5JFHHsmrr77a9CajefPmZZtttmnRPNtvv33T95VKJX379m063aS2tjaHH354rrzyyhxyyCF55JFHmo6aAySrfz5VKpU1lk+YMCGjR4/OkCFDMnbs2Oy///7Ze++9k6z+Obd48eJsuOGGze7z5ptv5umnn06SLFmyJOeee25uv/32PP/882loaMibb77ZdAR79OjRGThwYAYPHpyxY8dm7NixOeigg9K5c+c8/vjjWbZsWUaPHt1s/ytWrMiOO+6YJJk9e3Z22WWXZrOPGDGi3AvDh47Ahg+wLl26ZPPNN2+27Lnnnmv6vrGxMZ///Odz0kknrXHfAQMGZMmSJdl7772z995755prrkmvXr0yb968jBkzJitWrGjxPB06dGh2u1KpNLsqwMSJE7PDDjvkueeey5VXXpm99torAwcObPHjAB9Os2fPzqabbrrG8mHDhmXOnDm54447ctddd+WQQw7JqFGjcuONN6axsTH9+vXLjBkz1rjfW1c8Ou200/Lzn/88F154YTbffPN06tQpBx98cNPPuW7duuXhhx/OjBkz8otf/CJnnXVWzjnnnPzhD39o+hn2s5/9LJtsskmz/b/1JsxqtVrwVeCjQGBDGzZs2LA89thja0T4W2bNmpVXXnklF1xwQfr3759k9bmI76Rjx45ZtWrVe5pnu+22y/Dhw3PFFVfkuuuuy6WXXvqe9gN8+Nx9992ZNWtWTjnllLWur6+vz/jx4zN+/PgcfPDBGTt2bF599dUMGzYsL7zwQtq3b990XvbfuvfeezNhwoQcdNBBSVafk/23b0Js3759Ro0alVGjRuXss89Ojx49cvfdd2f06NGpra3NvHnzsscee6x1/9tss80af4174IEHWvT8+WgR2NCGnX766dlll13yhS98Icccc0y6dOmS2bNn55e//GUuvfTSDBgwIB07dsyll16a4447Ln/6059y3nnnveM+Bw0alMWLF+dXv/pVhg4dms6dO6dz587rPNPEiRNz4oknpnPnzk2/7ICPluXLl+eFF17IqlWr8uKLL+bOO+/MpEmTsv/+++dzn/vcGttffPHF6devX3bYYYfU1NTkxz/+cfr27ZsePXpk1KhRGTFiRMaNG5fJkydnyJAhef755zN9+vSMGzcuw4cPz+abb56bbropBxxwQCqVSr75zW82++va7bffnmeeeSa77757Nthgg0yfPj2NjY0ZMmRIunXrlq985Ss55ZRT0tjYmN122y2LFi3K/fffn65du+aII47Icccdl+9+97s59dRT8/nPfz4PPfRQ09VQYG1cRQTasO233z6//vWv85//+Z/55Cc/mR133DHf/OY3069fvyRJr169Mm3atPz4xz/ONttskwsuuCAXXnjhO+5z1113zXHHHZfx48enV69e+c53vtOimT772c+mffv2Oeyww1JXV/eenxvQdt15553p169fBg0alLFjx+aee+7JlClTcsstt6Rdu3ZrbN+1a9dMnjw5w4cPz0477ZS5c+dm+vTpqampSaVSyfTp07P77rvnqKOOypZbbplDDz00c+fOTZ8+fZKsDvQNNtggu+66aw444ICMGTMmw4YNa9p/jx49ctNNN2XkyJHZeuutc9lll+VHP/pRtt122yTJeeedl7POOiuTJk3K1ltvnTFjxuS2225rOp1lwIAB+clPfpLbbrstQ4cOzWWXXZbzzz//7/BK0lZVqk4sAgqaP39+Bg0alD/84Q/NfsEBwEeFwAaKWLlyZRYsWJAzzjgjzz77bO67777WHgkAWoVTRIAi7rvvvgwcODAPPfRQLrvsstYeBwBajSPYAABQkCPYAABQkMAGAICCBDYAABQksAEAoCCBDfARdM4552SHHXZouj1hwoSMGzfu7z7H3LlzU6lU8sgjj7ztNoMGDcoll1yyzvucNm1aevTo8b5nq1Qqa3w8NsC6ENgAHxATJkxIpVJJpVJJhw4dMnjw4HzlK1/JkiVL1vtjf+9731vnj35elygG+Chr39oDAPA/xo4dm6uuuiorV67Mvffem4kTJ2bJkiWZOnXqGtuuXLkyHTp0KPK43bt3L7IfABzBBvhAqa2tTd++fdO/f/8cdthh+ed//uem0xTeOq3jyiuvzODBg1NbW5tqtZqFCxfm2GOPTe/evVNfX5+RI0fm0UcfbbbfCy64IH369Em3bt1y9NFHZ9myZc3W/+0pIo2NjZk8eXI233zz1NbWZsCAAfn2t7+dJNl0002TJDvuuGMqlUo+9alPNd3vqquuytZbb526urpstdVW+f73v9/scX7/+99nxx13TF1dXYYPH56ZM2e2+DW66KKLst1226VLly7p379/TjjhhCxevHiN7W6++eZsueWWqaury+jRozN//vxm62+77bZ8/OMfT11dXQYPHpxzzz03DQ0Na33MFStW5MQTT0y/fv1SV1eXQYMGZdKkSS2eHfhocAQb4AOsU6dOWblyZdPt//qv/8oNN9yQn/zkJ2nXrl2SZL/99kvPnj0zffr0dO/ePZdffnn22muvPPXUU+nZs2duuOGGnH322fm3f/u3fPKTn8wPf/jDTJkyJYMHD37bx/3a176WK664IhdffHF22223LFiwIE888USS1ZG8884756677sq2226bjh07JkmuuOKKnH322fk//+f/ZMcdd8zMmTNzzDHHpEuXLjniiCOyZMmS7L///hk5cmSuueaazJkzJ1/60pda/JrU1NRkypQpGTRoUObMmZMTTjghX/3qV5vF/NKlS/Ptb387V199dTp27JgTTjghhx56aO67774kyc9//vMcfvjhmTJlSj75yU/m6aefzrHHHpskOfvss9d4zClTpuTWW2/NDTfckAEDBmT+/PlrBDtAkyoAHwhHHHFE9cADD2y6/bvf/a664YYbVg855JBqtVqtnn322dUOHTpUX3rppaZtfvWrX1Xr6+ury5Yta7avzTbbrHr55ZdXq9VqdcSIEdXjjjuu2fpPfOIT1aFDh671sRctWlStra2tXnHFFWudc86cOdUk1ZkzZzZb3r9//+p1113XbNl5551XHTFiRLVarVYvv/zyas+ePatLlixpWj916tS17uuvDRw4sHrxxRe/7fobbrihuuGGGzbdvuqqq6pJqg888EDTstmzZ1eTVH/3u99Vq9Vq9ZOf/GT1/PPPb7afH/7wh9V+/fo13U5S/elPf1qtVqvVL37xi9WRI0dWGxsb33YOgLc4gg3wAXL77bena9euaWhoyMqVK3PggQfm0ksvbVo/cODA9OrVq+n2Qw89lMWLF2fDDTdstp8333wzTz/9dJJk9uzZOe6445qtHzFiRO655561zjB79uwsX748e+211zrP/fLLL2f+/Pk5+uijc8wxxzQtb2hoaDq/e/bs2Rk6dGg6d+7cbI6Wuueee3L++efn8ccfz6JFi9LQ0JBly5ZlyZIl6dKlS5Kkffv2GT58eNN9ttpqq/To0SOzZ8/OzjvvnIceeih/+MMfmk57SZJVq1Zl2bJlWbp0abMZk9Wn0IwePTpDhgzJ2LFjs//++2fvvfdu8ezAR4PABvgA2XPPPTN16tR06NAhG2+88RpvYnwrIN/S2NiYfv36ZcaMGWvs671eqq5Tp04tvk9jY2OS1aeJfOITn2i27q1TWarV6nua5689++yz2XfffXPcccflvPPOS8+ePfOb3/wmRx99dLNTaZLVl9n7W28ta2xszLnnnptPf/rTa2xTV1e3xrJhw4Zlzpw5ueOOO3LXXXflkEMOyahRo3LjjTe+7+cEfPgIbIAPkC5dumTzzTdf5+2HDRuWF154Ie3bt8+gQYPWus3WW2+dBx54IJ/73Oealj3wwANvu88tttginTp1yq9+9atMnDhxjfVvnXO9atWqpmV9+vTJJptskmeeeSb//M//vNb9brPNNvnhD3+YN998syni32mOtXnwwQfT0NCQ7373u6mpWf0+/RtuuGGN7RoaGvLggw9m5513TpI8+eSTef3117PVVlslWf26Pfnkky16revr6zN+/PiMHz8+Bx98cMaOHZtXX301PXv2bNFzAD78BDZAGzZq1KiMGDEi48aNy+TJkzNkyJA8//zzmT59esaNG5fhw4fnS1/6Uo444ogMHz48u+22W6699to89thjb/smx7q6upx++un56le/mo4dO+Yf//Ef8/LLL+exxx7L0Ucfnd69e6dTp065884787GPfSx1dXXp3r17zjnnnJx00kmpr6/PPvvsk+XLl+fBBx/Ma6+9llNPPTWHHXZYzjzzzBx99NH5xje+kblz5+bCCy9s0fPdbLPN0tDQkEsvvTQHHHBA7rvvvlx22WVrbNehQ4d88YtfzJQpU9KhQ4eceOKJ2WWXXZqC+6yzzsr++++f/v375zOf+Uxqamryxz/+MbNmzcq3vvWtNfZ38cUXp1+/ftlhhx1SU1OTH//4x+nbt2+RD7QBPnxcpg+gDatUKpk+fXp23333HHXUUdlyyy1z6KGHZu7cuenTp0+SZPz48TnrrLNy+umn5+Mf/3ieffbZHH/88e+4329+85v58pe/nLPOOitbb711xo8fn5deeinJ6vObp0yZkssvvzwbb7xxDjzwwCTJxIkT84Mf/CDTpk3Ldtttlz322CPTpk1ruqxf165dc9ttt+Xxxx/PjjvumDPPPDOTJ09u0fPdYYcdctFFF2Xy5Mn5h3/4h1x77bVrvVxe586dc/rpp+ewww7LiBEj0qlTp/zf//t/m9aPGTMmt99+e375y19mp512yi677JKLLrooAwcOXOvjdu3aNZMnT87w4cOz0047Ze7cuZk+fXrTUXSAv1apljgpDgAASOIINgAAFCWwAQCgIIENAAAFCWwAAChIYAMAQEECGwAAChLYAABQkMAGAICCBDYAABQksAEAoCCBDQAABQlsAAAo6P8DF1uUuzGlEIkAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9366827253957329\n",
      "Precision: 1.0\n",
      "Recall: 0.9350741002117149\n",
      "F1-score: 0.9664478482859227\n",
      "Sensitivity: 0.9350741002117149\n",
      "Specificity: 1.0\n",
      "True positive rate: 0.9350741002117149\n",
      "False positive rate: 0.0\n",
      "True negative rate: 1.0\n",
      "False negative rate: 0.06492589978828511\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.28      1.00      0.44        36\n",
      "           1       1.00      0.94      0.97      1417\n",
      "\n",
      "    accuracy                           0.94      1453\n",
      "   macro avg       0.64      0.97      0.70      1453\n",
      "weighted avg       0.98      0.94      0.95      1453\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Get confusion matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "# Plot confusion matrix as heatmap\n",
    "fig, ax = plt.subplots(figsize=(8,8))\n",
    "ax.imshow(cm)\n",
    "ax.grid(False)\n",
    "ax.set_xlabel('Predicted labels')\n",
    "ax.set_ylabel('True labels')\n",
    "ax.xaxis.set(ticks=(0, 1), ticklabels=('Healthy', 'Diseased'))\n",
    "ax.yaxis.set(ticks=(0, 1), ticklabels=('Healthy', 'Diseased'))\n",
    "for i in range(2):\n",
    "    for j in range(2):\n",
    "        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')\n",
    "plt.show()\n",
    "tn, fp, fn, tp = cm.ravel()\n",
    "\n",
    "# Calculate various evaluation metrics\n",
    "accuracy = (tp + tn) / (tp + tn + fp + fn)\n",
    "precision = tp / (tp + fp)\n",
    "recall = tp / (tp + fn)\n",
    "f1_score = 2 * precision * recall / (precision + recall)\n",
    "sensitivity = tp / (tp + fn)\n",
    "specificity = tn / (tn + fp)\n",
    "true_positive_rate = tp / (tp + fn)\n",
    "false_positive_rate = fp / (tn + fp)\n",
    "true_negative_rate = tn / (tn + fp)\n",
    "false_negative_rate = fn / (tp + fn)\n",
    "\n",
    "# Print the evaluation metrics\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Precision:\", precision)\n",
    "print(\"Recall:\", recall)\n",
    "print(\"F1-score:\", f1_score)\n",
    "print(\"Sensitivity:\", sensitivity)\n",
    "print(\"Specificity:\", specificity)\n",
    "print(\"True positive rate:\", true_positive_rate)\n",
    "print(\"False positive rate:\", false_positive_rate)\n",
    "print(\"True negative rate:\", true_negative_rate)\n",
    "print(\"False negative rate:\", false_negative_rate)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a95165b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
