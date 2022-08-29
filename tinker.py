import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import torch
if __name__ == "__main__":

    # y = np.zeros(50)
    # y_hat = np.zeros(50)

    reference = Annotation(uri='file1')
    reference[Segment(0, 55)] = 'A'
    reference[Segment(55, 100)] = 'B'
    
    hypothesis = Annotation(uri='file1')
    hypothesis[Segment(0, 55)] = 'a'
    hypothesis[Segment(54, 74)] = 'b'
    hypothesis[Segment(75, 100)] = 'c'
    
    metric = DiarizationErrorRate(collar=5)

    print(abs(metric(reference, hypothesis)))

    unbatched_labels=[torch.ones(50,), torch.zeros(50,)]
    unbatched_preds =[torch.cat([torch.zeros(25,), torch.ones(25,)]), torch.cat([torch.zeros(25,), torch.ones(25,)])]

    references = []
    hypothesises = []
    label_key =  {0:'ZERO', 1:'ONE', 2:'TWO', 3:'THREE', 4:'FOUR'}
    for pred, label in zip(unbatched_preds, unbatched_labels):
        

        reference = Annotation(uri='file1')
        hypothesis = Annotation(uri='file1')

        grad_pred = torch.gradient(pred, dim=-1)[0]
        switches_pred = torch.where(grad_pred!=0)

        pred = pred.numpy()

        if len(switches_pred[0]) == 0: 
            hypothesis[Segment(0, len(pred))] = label_key[pred[0]].lower()

        else:

            l_index = switches_pred[0][0::2].numpy()
            r_index = switches_pred[0][1::2].numpy()

            switch_count = 0
            n_switches = len(r_index)
            for l, r in zip(l_index, r_index):

                if switch_count == 0: 
                    hypothesis[Segment(0, l)] = label_key[pred[l]].lower()
                    r_prev = r
                else:
                    hypothesis[Segment(r_prev, l)] = label_key[pred[l]].lower()
                    r_prev = r

                switch_count+= 1

                if switch_count == n_switches:
                    hypothesis[Segment(r_prev, len(pred)-1)] = label_key[pred[len(pred)-1]].lower()

        grad_label = torch.gradient(label, dim=-1)[0]
        switches_label = torch.where(grad_label!=0)
        label = label.numpy()

        if len(switches_label[0]) == 0: 
            reference[Segment(0, len(label))] = label_key[label[0]]
            print(reference)

        else:
            l_index = switches_label[0][0::2].numpy()
            r_index = switches_label[0][1::2].numpy()

            switch_count = 0
            n_switches = len(r_index)
            for l, r in zip(l_index, r_index):

                if switch_count == 0: 
                    reference[Segment(0, l)] = label_key[label[l]]
                    r_prev = r
                else:
                    reference[Segment(r_prev, l)] = label_key[label[l]]
                    r_prev = r

                switch_count+= 1

                if switch_count == n_switches:
                    reference[Segment(r_prev, len(label)-1)] = label_key[label[len(label)-1]]

        references.append(reference)
        hypothesises.append(hypothesis)

    metric = DiarizationErrorRate(collar=0)

    for reference, hypothesis in zip(references, hypothesises):  
        metric(reference, hypothesis)      
    global_value = abs(metric)            
    mean, (lower, upper) = metric.confidence_interval() 

    print(metric)
    print(f'DER: {global_value}')
    print(f'Mean DER: {mean}')