'''
This utility takes a set of tar files (each of which contains a set of images) and creates an HDF5 file
for each input tar file. The HDF5  file contains 2 datasets:
 "filenames": the input filenames that created the features
 "feats": The extracted features

 Notes:
   - Currently supports multiple GPUs
   - Requires Caffe/pycaffe be installed and in the path.
'''
import pickle
import glob
import bz2
import os

import numpy as np
import h5py
import multiprocessing
import subprocess
import tarfile
import time
import shutil

# NOTE: These need to point at the appropriate location  for your setup
DEFAULT_IMAGE = '/data/yfcc_images/10000029263_4b1105fbf4.jpg'

# These two files can be downloaded from the following locations
# PATH_MODEL_DEF: https://raw.githubusercontent.com/karpathy/neuraltalk/master/python_features/deploy_features.prototxt
# PATH_MODEL: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
PATH_MODEL_DEF = '/data/model/deploy_features.prototxt' #
PATH_MODEL = '/data/model/VGG_ILSVRC_16_layers.caffemodel'

working_dir = "/working"


def caffe_extract_feats(caffe_net, path_imgs, batch_size = 10 ):
    '''
    Function using the caffe python wrapper to extract 4096 from VGG_ILSVRC_16_layers.caffemodel model

    Inputs:
    ------
    path_imgs      : list of the full path of images to be processed
    path_model_def : path to the model definition file
    path_model     : path to the pretrained model weight
    WItH_GPU       : Use a GPU

    Output:
    -------
    features           : return the features extracted
    '''
    # Use local imports here to play nice with multiprocessing
    import caffe
    from py_caffe_feat_extract import preprocess_image
    feats = np.zeros((4096 , len(path_imgs)))

    for b in range(0 , len(path_imgs) , batch_size):
        try:
            # Create one batch of images at a time
            list_imgs = []
            for i in range(b , b + batch_size ):
                if i < len(path_imgs):
                    try:
                        list_imgs.append( np.array( caffe.io.load_image(path_imgs[i]) ) ) #loading images HxWx3 (RGB)
                    except:
                        # TODO: Perhaps there is a better option than just adding a "default" image if we can't load
                        #        the image we are supposed to load
                        list_imgs.append( np.array( caffe.io.load_image(DEFAULT_IMAGE) ) ) #loading images HxWx3 (RGB)
                else:
                    list_imgs.append(list_imgs[-1]) #Appending the last image in order to have a batch of size 10. The extra predictions are removed later..

            # Preprocess and run through our network
            caffe_input = np.asarray([preprocess_image(in_) for in_ in list_imgs]) #preprocess the images
            start_time = time.time()
            predictions =caffe_net.forward(data = caffe_input)
            print 'Elapsed Time: ', time.time() - start_time

            # Add to output
            predictions = predictions[caffe_net.outputs[0]].transpose()
            if i < len(path_imgs):
                feats[:,b:i+1] = predictions
                n = i+1
            else:
                n = min(batch_size , len(path_imgs) - b)
                feats[:,b:b+n] = predictions[:,0:n] #Removing extra predictions, due to the extra last image appending.
                n += b
            print "%d out of %d done....."%(n ,len(path_imgs))
        except:
            raise
    return feats


def save_hdf5(local_working_dir, hdf5_fname, image_features, im_files_for_batch):
    '''
    Create hdf5 file from features and filename list
    '''
    bname = os.path.basename(hdf5_fname)
    temp_fname = os.path.join(local_working_dir, bname)

    im_files_wo_fnames = [os.path.basename(file) for file in im_files_for_batch]
    fOut = h5py.File(temp_fname, 'w')
    fOut.create_dataset('filenames', data=im_files_wo_fnames)
    fOut.create_dataset('feats', data=image_features, dtype=np.float32, compression='gzip')
    fOut.close()

    shutil.move(temp_fname, hdf5_fname)



def tar_file_worker(work_queue, result_queue):
    import caffe

    # NOTE: If you do not have a GPU in your box then use caffe.set_mode_cpu() and don't bother setting a device
    caffe.set_mode_gpu()
    try:
        caffe.set_device(int(multiprocessing.current_process().name[-1]) -1)
    except:
        caffe.set_device(0)

    # Load our network parameters
    caffe_net = caffe.Classifier(PATH_MODEL_DEF , PATH_MODEL , image_dims = (224,224) , raw_scale = 255, channel_swap=(2,1,0),
                               mean = np.array([103.939, 116.779, 123.68]) )


    # Create a working directory for this multiprocessing worker to untar files to
    local_working_dir = os.path.join(working_dir, multiprocessing.current_process().name)
    if not os.path.exists(local_working_dir):
        if not os.path.isdir(os.path.dirname(local_working_dir)):
            os.mkdir(os.path.dirname(local_working_dir))
        os.mkdir(local_working_dir)

    for item in iter(work_queue.get, 'STOP'):
        # Iterate through work_queue processing a tar file at a time
        try:
            print 'Clearing local working dir'
            cmd = ['rm', '-rf',local_working_dir + '/*']
            p =  subprocess.Popen(cmd)

            (tarfile_name, pickle_fname) = item
            # Untar file to working dir
            print 'Untarring:', tarfile_name
            cmd = ['tar', '-xf', tarfile_name, '-C', local_working_dir]
            p =  subprocess.Popen(cmd)
            print 'Extracting Batch and writing', pickle_fname
            im_files_for_batch = glob.glob(os.path.join(local_working_dir, '*'))
            image_features = caffe_extract_feats(caffe_net, im_files_for_batch, 10)
            save_hdf5(local_working_dir, pickle_fname, image_features, im_files_for_batch)

        except:
            raise

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--tarfile_dir", dest="tarfile_dir", help="Directory containing tar files")
    parser.add_option("--hdf5_dir", dest="hdf5_dir", help="Directory to put hdf5 files in")
    (options, args) = parser.parse_args()
    num_workers =1 # Should be set to # of GPUs
    work_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    tarfile_dir = options.tarfile_dir
    output_dir = options.hdf5_dir
    tarfiles = sorted(glob.glob(os.path.join(tarfile_dir, '*.tar')), key=os.path.getsize, reverse=False)

    hdf5_files = set(glob.glob(os.path.join(output_dir, '*')))
    # Add tar files in tarfile_dir to queue taking care to not include files we've already processed
    for tarfile_name in tarfiles:
        bname = os.path.basename(tarfile_name)
        output_fname = os.path.join(output_dir, bname.split('.')[0] + '.hdf5')
        if output_fname not in hdf5_files:
            #print tarfile_name, output_fname
            work_queue.put((tarfile_name, output_fname))

    workers =[]
    for w in range(num_workers):
        p = multiprocessing.Process(target=tar_file_worker, args=(work_queue, result_queue))
        p.start()
        workers.append(p)
        work_queue.put('STOP')

    for p in workers:
        p.join()

if __name__ == "__main__":
    main()