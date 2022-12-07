'''
This file is used to show that all modules work as expected
'''
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import easyocr
import utils

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

from PIL import Image
from collections import OrderedDict

# PATHS: 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_DIR = 'customTF2/training/MyMobileNet'
CONFIG_DIR = config_util.get_configs_from_pipeline_file('customTF2/models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/pipeline.config')
MODEL_CONFIG = CONFIG_DIR['model']
TEST_IMAGE_PATHS = glob.glob('customTF2/data/test_labels_jpg/*.jpg')

def get_img_paths(parameters):
    img_path, K = parameters.parameters_txt.readline().strip(), int(parameters.parameters_txt.readline())
    directory = os.getcwd() + '/' + parameters.parameters_txt.name.split('/')[0] + '/' + img_path
    print(directory)
    print(img_path)
    paths = next(os.walk(directory), (None, None, []))[2]
    

    paths = [parameters.parameters_txt.name.split('/')[0] + '/' + img_path + '/' + path for path in paths]
    # print(sorted(paths))
    return sorted(paths), K

def ODM(image_path): #A function for conducting object detection (Module 1)
    detection_model = model_builder.build(model_config=MODEL_CONFIG, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore('customTF2/training/MyMobileNet/ckpt-6.index'.replace('.index', '')).expect_partial()
    detect_fn = utils.get_model_detection_function(detection_model)

    label_map_path = CONFIG_DIR['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    image_np = utils.load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()


    idx, soft_scores = tf.image.non_max_suppression_with_scores(detections['detection_boxes'][0], detections['detection_scores'][0], max_output_size=len(detections['detection_boxes'][0]), iou_threshold=.99, score_threshold=.25, soft_nms_sigma=.2)

    boxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    scores = detections['detection_scores'][0].numpy()

    selected_boxes = boxes[idx.numpy()]
    selected_classes = classes[idx.numpy()]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        selected_boxes,
        classes,
        soft_scores.numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.25,
        agnostic_mode=False,
    )
    plt.figure(figsize=(20,20))
    plt.imshow(image_np_with_detections)
    path = os.getcwd() + '/detections/'
    if '.png' in image_path:
        path += image_path.replace('.png', '')
    elif '.jpeg' in image_path:
        path += image_path.replace('.jpeg', '')
    else:
        path += image_path.replace('.jpg', '')
    
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/' + image_path.split('/')[-1]
    plt.savefig(path)
    plt.close()
    print('Done')
    return path, detections, idx, categories

def OCRM(image_path, detections, idx, categories): # A function for conducting Character Recognition (Module 2)
    img = Image.open(image_path)
    label_id_offset = 1
    boxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)

    selected_boxes = utils.normalized_coordinates(img, boxes[idx.numpy()]) # ymin, xmin, ymax, xmax
    selected_classes = classes[idx.numpy()]
    i = 0
    results = []
    for box, label in zip(selected_boxes, selected_classes):
        print(f"box {i}: {box}")
        curr_label = ""
        for category in categories:
            if category['id'] == label:
                curr_label = category['name']
                break
        
        curr_result = [curr_label]
        temp_img = img
        crop_rectangle = (box[1], box[0], box[3], box[2]) # left, upper, right, lower
        cropped_im = np.array(temp_img.crop(crop_rectangle))
        reader = easyocr.Reader(['en'], gpu=True)
        result = reader.readtext(cropped_im)
        for r in result:
            curr_result.append(r[1])

        results.append(curr_result)

        i += 1
    print('Done')
    return results

# a function for preprocessing the OCR (Module 3)
def preprocess(words=None):
    #words = [['ident_rel', 'Contains'], ['ident_rel', 'By'], ['rel', 'stion 1'], ['weak_entity', 'PK', 'trip_number', 'type', 'maximum_weight', 'price', '2500', '500', '1000', 'bag_'], ['ident_rel'], ['entity'], ['rel_attr', 'Sold By'], ['weak_entity', 'PK', 'trip_number', 'is_second_flight', 'airlint_name', 'depature_', 'date_time', 'arrival_date_time', 'duration', '2000', '2500', '3000'], ['entity', 'Trip', 'PK', 'trip_number', 'class', 'price', 'origin', 'destination'], ['many'], ['one'], ['many'], ['one'], ['many'], ['many'], ['one']] # test string to preprocess
    
    sb = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    filtered_words = []
    for word_list in words:
        if 'ident_rel' not in word_list and 'rel' not in word_list and 'one' not in word_list and 'many' not in word_list: # removes relationships as they are not needed in our application
            filtered_words.extend(word_list)
    

    filtered_words = ' '.join(wnl.lemmatize(sb.stem(word.lower().replace('_', ' ').strip())) for word in filtered_words if (not word.isnumeric()) and word not in set(stopwords.words('english'))) # removed one and many as our model failed to classify them reliably
    # print(filtered_words)
    return filtered_words

# Function used from here: https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

# a function for finding optimal number of clusters using silhouette scoring
def silhouette(X):
    sil_scores = []

    for k in range(2, X.shape[0]):
        cluster = KMeans(n_clusters=k, random_state=42)
        cluster_labels = cluster.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg,)
        sil_scores.append((k, silhouette_avg))
    
    optimal_k = max(sil_scores, key = lambda x: x[1])[0]
    # print(optimal_k)
    return optimal_k

# a function for conducting Kmeans
def baseKMeans(K, imgs, documents, filename, mode=1):
    # documents = ['rel attr weak ent flight nation rture tim rel attr rel attr rel attr entiti entiti entiti two-way trip pk number class price book bboking airlin entiti', 'rel attr rel attr entiti luuu entiti weak ent entiti airlin name rate weak ent de de', 'entiti bag rel attr entiti rel attr entiti round trip id inat imum bag arture datetime to al datetim tion to arture datetime from al datetime from tion datetime from entiti one way trip pk trip id class price origin destin maximum bag departure datetim arrival datetim durat', 'weak ent flight pk origin destin departure date-tim arrival date-tim entiti weak ent two-way trip k number class entiti pk name rate weak ent one-way trip pk number class rel attr rel attr rel attr rel attr', 'weak ent round-t pk number class price origin destin maximum nun weak ent one-way pk number class price origin destin departure d arrival date-tin durat maximum num entiti flight flight numb departure date-tim arrival date-tim durat entiti ig typ aximum weight entiti rel attr rel attr', 'rel attr entiti flight liahtnumb rel attr rel attr entiti eutit trip entiti rate weak ent numtyp maximumweight', 'weak ent flight pk,fk number origin departure date tim destination arrival date tim destination arrival date tim origin to destination dur destination departure date time nul origin arrivial date time nul destin to origin duration nul comment: in q1, making on table for flight: return flight entiti entiti trip pk number type class price entiti rel attr maximum   numb rel attr', 'entiti pe nam aximum weight flight llight numb departure date tim arriv date time durat rel attr entiti entiti entiti trip pk trip numb class origin price', 'entiti rel attr entiti entiti trip pk uid type class origin destin price entiti flig pk flight uid origin destin departure-- arrival-tim durat', 'entiti pk name rate rel attr rel attr rel attr rel attr entiti trip pk number class price type rel attr entiti weight bag rel attr rel attr', 'rel attr entiti "9 entiti bag pe ax weight entiti trip pk uniqueld class price rel attr entiti one-w pk uniqueld entiti entiti']
    
    if mode==1:
        count_vectorizer = CountVectorizer()
        a = count_vectorizer.fit_transform(documents).toarray()
        X = np.where(a != 0, np.log10(a) + 1, 0) # apply log10(tf) + 1
        X_std = StandardScaler().fit_transform(X)
        #print(X_std.shape)
        lsa = make_pipeline(TruncatedSVD(n_components=5), Normalizer(copy=False)) # eliminating unneccessary dimensions
        X_lsa = lsa.fit_transform(X_std)
        #print(X_lsa.shape)
    elif mode==2:
        documents = [sent.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~')) for sent in documents]
        tokenized_docs = [nltk.word_tokenize(doc) for doc in documents]

        model = FastText(vector_size=50, window=2, min_count=1, sentences=tokenized_docs, epochs=1000, seed=42) 

        vectorized_docs = vectorize(tokenized_docs, model=model)
        X_lsa = vectorized_docs
        lsa = make_pipeline(TruncatedSVD(n_components=10), Normalizer(copy=False)) # eliminating unneccessary dimensions

    sil_K = 0
    if K == 0:
        print('conducting silhouette clustering')
        K = silhouette(X_lsa)
        sil_K = K 

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=1000)
    kmeans.fit_predict(X_lsa)
    clusters = kmeans.labels_ + 1
    


    imgs = [img.split('/')[-1] for img in imgs]

    pred_cluster_dict = OrderedDict()
    for c, img in zip(clusters, imgs):
        if 'C' + str(c) not in pred_cluster_dict:
            pred_cluster_dict['C' + str(c)] = [img]
        else:
            pred_cluster_dict['C' + str(c)].append(img)

    #rs = rand_score(gtc, clusters)
    with open(filename, 'w') as f:
        for k in sorted(pred_cluster_dict.keys()):
            f.write(f"{k}: {pred_cluster_dict[k]}")
            f.write('\n')
        
        #f.write(f"rand index: {str(rs)}")

    return sil_K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters_txt', type=argparse.FileType('r'), required=True, help='Text file containing ERD Image paths')
    #parser.add_argument('--rand_score', default=False, action="store_true", help='Flag used for determining rather or not to use rand score')
    parameters = parser.parse_args()
   # print(parameters.rand_score)
    print("getting ground_truth clusters")
    imgs, K = get_img_paths(parameters)

    documents = []
    for img in imgs:

        print(f'Conducting Object Detection on {img}')
        path, detections, idx, categories = ODM(img)
        print(f'Conducting Optical Character Recognition on {img}')
        results = OCRM(path, detections, idx, categories)
        print(f'Preprocessing Vocabulary for text from {img}')
        documents.append(preprocess(results))
    print('Finished extracting text from all images')



    print('Conducting K-Means++ on Documents (Module 4)')
    
    sil_K = baseKMeans(K, imgs, documents, filename='module_4.txt', mode=1) # utilizing module 4 to find optimal # of clusters
    print('Conducting K-Means++ using FastText embeddings (Module 5)')
    if sil_K > 0:
        K = sil_K
    baseKMeans(K, imgs, documents, filename='module_5.txt', mode=2)



if __name__ == "__main__":
    main()