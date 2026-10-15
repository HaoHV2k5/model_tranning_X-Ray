import argParser as ARG
ARG.initArgs() #init arguments before loading other modules

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from ChexpertDataset import ClassificationChexpertDataset
from NetworkModel import getModelClass
from tqdm import tqdm
import torchmetrics.functional as tmf

try:
    import wandb
except ImportError:
    wandb = None

import multiprocessing as mp

def initWandB():

    wandb.login()
    runVar = wandb.init(
        project="MEDICAL_PATCH_NET",
        config={
            "learning_rate": ARG.LEARNING_RATE,
            "epoch_num":ARG.EPOCH_NUM,
            "batch_size":ARG.BATCH_SIZE,
            "patch_size":ARG.PATCH_SIZE,
            "wandb_name":ARG.WANDB_NAME
        },
        name=ARG.WANDB_NAME
        )


def log(name,val,printLog=True,commit=False):
    if not printLog and not (ARG.USE_WANDB and wandb):
        return
    if isinstance(val,torch.Tensor): val = val.detach().cpu().item()
    if printLog: print("LOG",name,val)
    if ARG.USE_WANDB and wandb: wandb.log({name:val},commit=commit)

DEFAULT_MODEL_PATH = "savedModels/"

TEST = "test"

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        trainLoader: DataLoader,
        validLoaderList: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        gpuId,
        saveEvery: int = 1,
    ) -> None:
        
        self.gpuId = gpuId
        self.model = model.to(gpuId)
        self.trainLoader = trainLoader
        self.validLoaderList = validLoaderList
        self.optimizer = optimizer
        self.saveEvery = saveEvery
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.ampScaler = None

    def _runBatch(self, img, label):
        self.optimizer.zero_grad()

        output = self.model(img)
        loss = self.criterion(output,label)
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()
        
        log("learningRate",self.scheduler.get_last_lr()[0],printLog=False)
        log("loss",loss,commit=True,printLog=False)

    def _runEpoch(self, epoch):
        self.model.train()
        stepNum = len(self.trainLoader)
        print(f"Epoch: {epoch} | Batchsize: {ARG.BATCH_SIZE} | Steps: {stepNum}")
        
        for i, (img,label) in enumerate(tqdm(self.trainLoader, desc=f"Train Epoch {epoch}")):
            img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)
            label = label.to(self.gpuId,non_blocking=True)
            self._runBatch(img,label)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        for el in self.validLoaderList:
            if el[2] == TEST and (epoch != ARG.EPOCH_NUM - 1): continue #if a test set is given, only apply it to the last 
            self.validate(*el)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _saveCheckpoint(self, epoch):
        saveName = DEFAULT_MODEL_PATH + ARG.MODEL_SAVE_PATH
        torch.save(self.model, saveName) # save entire model
        torch.save(self.model.state_dict(), saveName.replace(".pt","_weights.pt")) # save weights only

        print(f"Epoch {epoch} | Checkpoint saved at {saveName}")

    def train(self, maxEpochs: int):
        for epoch in range(maxEpochs):
            log("epoch",epoch)
            self._runEpoch(epoch)
            if epoch % self.saveEvery == 0:
                self._saveCheckpoint(epoch)
        if ARG.USE_WANDB: wandb.log({},commit=True)
    
    def calculateThresholds(self, thresholdLoader):
        #this function optimizes the threshold based on the validation data, NOT on the test data, to prevent data leakage 
        self.model.eval()
        outputList = list()
        labelList = list()
        with torch.inference_mode():
            for img,label in tqdm(thresholdLoader,desc="optThreshold"):
                img = img.to(self.gpuId, non_blocking=True, dtype=torch.float)
                output = self.model(img)
                outputList.append(output.cpu())
                labelList.append(label.cpu())

        allOutput = torch.sigmoid(torch.cat(outputList).to(dtype=torch.float32))
        allLabel = torch.cat(labelList).to(dtype=torch.int32)
        labelNameList = thresholdLoader.dataset.getLabelNames()
        thrsDict = dict()

        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:, i]
            targets = allLabel[:, i]
            fpr, tpr, thresh = tmf.roc(preds, targets, task="binary")
            optIdx = torch.argmax(tpr - fpr)
            thrsDict[labelName] = thresh[optIdx].item()

        print(thrsDict)
        self.model.train()
        return thrsDict

    
    def validate(self,dataLoader,datasetName,validType,thresholdValidLoader=None):
        self.model.eval()
        outputList = list()
        labelList = list()
        with torch.inference_mode():
            for img,label in tqdm(dataLoader, desc=f"Validate {datasetName}"):
                img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)
                output = self.model(img)
                outputList.append(output.cpu())
                labelList.append(label.cpu())
        
        allOutput = torch.sigmoid(torch.cat(outputList).to(dtype=torch.float32)) # apply sigmoid to logits
        allLabel = torch.cat(labelList).to(dtype=torch.int32)

        labelNameList = dataLoader.dataset.getLabelNames()

        avgDict = dict()
        #calculate threshold based on validation data, not based on training data
        if thresholdValidLoader is not None: thrsDict = self.calculateThresholds(thresholdValidLoader)
        print("LEN:",len(labelNameList),labelNameList,allOutput.size(),allLabel.size())
        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:,i]
            targets = allLabel[:,i]

            def execMetric(func,name=None):
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary")
                log(datasetName+"/"+labelName + "/" + metricName,metr)
                if metricName not in avgDict: avgDict[metricName] = list()
                avgDict[metricName].append(metr)
            
            def execThrsMetric(func,threshold,name=None,execWithoutThreshold=True):
                if execWithoutThreshold: execMetric(func,name)
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary",threshold=threshold)
                log(datasetName+"/"+labelName + "/thrs_" + metricName,metr)
            
            def execBootstrapMetric(func,name=None,threshold=None,bootstrapNum = 1000):
                # Fast in-process bootstrap on Mac (avoids broken/slow mp.Pool spawn)
                valCount = len(preds)
                metrList = []
                for _ in range(bootstrapNum):
                    sampled_idx = torch.randint(0, valCount, (valCount,))
                    s_preds = preds[sampled_idx]
                    s_targets = targets[sampled_idx]
                    if threshold is None and s_targets.min() == s_targets.max():
                        continue
                    try:
                        if threshold is None:
                            m = func(s_preds, s_targets, task="binary").item()
                        else:
                            m = func(s_preds, s_targets, task="binary", threshold=threshold).item()
                        metrList.append(m)
                    except Exception:
                        continue

                if not metrList: return
                metrList.sort()
                n = len(metrList)
                lowConfIdx = min(round(n*0.025), n - 1)
                highConfIdx = min(round(n*0.975), n - 1)
                medianIdx = min(round(n*0.5), n - 1)

                meanVal = sum(metrList)/n
                metricName = func.__name__ if name is None else name
                thrsTag = "" if threshold is None else "thrs_"
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_low_" + metricName,metrList[lowConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_high_" + metricName,metrList[highConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_median_" + metricName,metrList[medianIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_mean_" + metricName,meanVal)

            execMetric(tmf.auroc)

            optThrs = 0.5
            if thresholdValidLoader is not None:
                optThrs = thrsDict[labelName]
                log(datasetName+"/"+labelName+"/opt_thrs",optThrs)
                
                execThrsMetric(tmf.accuracy,threshold=optThrs)
                execThrsMetric(tmf.precision,threshold=optThrs)
                execThrsMetric(tmf.recall,threshold=optThrs,name="sensitivity")
                execThrsMetric(tmf.specificity,threshold=optThrs)
                execThrsMetric(tmf.f1_score,threshold=optThrs)
            
            if validType == TEST:
                execBootstrapMetric(tmf.auroc)
                execBootstrapBoth = lambda func,name=None: (execBootstrapMetric(func,name=name),execBootstrapMetric(func,name=name,threshold=optThrs))
                execBootstrapBoth(tmf.accuracy)
                execBootstrapBoth(tmf.precision)
                execBootstrapBoth(tmf.recall,name="sensitivity")
                execBootstrapBoth(tmf.specificity)
                execBootstrapBoth(tmf.f1_score)


        for key in avgDict.keys():
            valList = avgDict[key]
            avgVal = sum(valList)/len(valList)
            log(datasetName+"/avg/"+key,avgVal)

        self.model.train()


def prepareDataloader(dataset: Dataset, batchSize: int, training: bool):
    num_proc = ARG.PROC_NUM if ARG.PROC_NUM is not None else 2
    kwargs = {
        "batch_size": batchSize,
        "shuffle": training,
        "drop_last": training,
        "num_workers": num_proc,
        "pin_memory": False,
    }
    if num_proc > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)

def loadModelFromCheckpoint(weightPath):
    ModelClass = getModelClass(ARG.MODEL_CLASS)
    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)
    loadedWeights = torch.load(weightPath,weights_only=False)
    loadedWeights = {k.replace("_orig_mod.",""):v for k,v in loadedWeights.items()}
    model.load_state_dict(loadedWeights)
    return model

def loadTrainObjs():
    trainSet = ClassificationChexpertDataset("train",augementImg=True)


    ModelClass = getModelClass(ARG.MODEL_CLASS)
    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)

    optimizer = torch.optim.AdamW(model.parameters(),lr=ARG.LEARNING_RATE)
    return trainSet, model, optimizer

def loadValidationLoaderList():
    validLoaderList = list()

    chexpertValidationSet = ClassificationChexpertDataset("validate",False)
    chexpertValidLoader = prepareDataloader(chexpertValidationSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertValidLoader,"val_Chexpert", "val", chexpertValidLoader))


    ########
    # This section adding the test set was added after the hyperparameter optimization on the validation set.
    # We added it for the final runs to then directly get the test results.
    # During testing different achitectures and approaches this was not part of the code and the test dataset was not touched
    chexpertTestSet = ClassificationChexpertDataset(TEST,False)
    chexpertTestLoader = prepareDataloader(chexpertTestSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertTestLoader,"test_Chexpert", TEST, chexpertValidLoader)) # the chexpertValidLoader is used to calculate the optimal threshold based on the validation set
    #######

    return validLoaderList




def trainAndEval(device, totalEpochs, saveEvery, batchSize):
    trainSet, model, optimizer = loadTrainObjs()
    trainLoader = prepareDataloader(trainSet, batchSize, training=True)
    #pretrainLoader = prepareDataloader(pretrainSet, batchSize, training=True) if pretrainSet is not None else None
    validLoaderList = loadValidationLoaderList()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,total_steps=ARG.EPOCH_NUM*len(trainLoader),max_lr=ARG.LEARNING_RATE,pct_start=0.05)

    trainer = Trainer(
        model=model,
        trainLoader=trainLoader,
        validLoaderList=validLoaderList,
        optimizer=optimizer,
        scheduler=scheduler,
        gpuId=device,
        saveEvery=saveEvery,
        #validsPerEpoch=ARG.VALIDS_PER_EPOCH,
    )
    trainer.train(totalEpochs)

def evalOnly(device,modelPath):
    validLoaderList = loadValidationLoaderList()
    model = loadModelFromCheckpoint(modelPath)
    trainer = Trainer(
        model=model,
        trainLoader=None,
        validLoaderList=None,
        optimizer=None,
        scheduler=None,
        gpuId=device,
        saveEvery=-1,
        #validsPerEpoch=ARG.VALIDS_PER_EPOCH,
    )
    for el in validLoaderList:
        trainer.validate(*el)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    if ARG.USE_WANDB: initWandB()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Using device:', device)
    if ARG.EVAL_ONLY:
        assert ARG.MODEL_LOAD_PATH is not None, "provide a path to model weights"
        evalOnly(
            device=device,
            modelPath=ARG.MODEL_LOAD_PATH,
        )
    else:
        assert ARG.MODEL_LOAD_PATH is None, "continue training not implemented yet."
        trainAndEval(
            device=device,
            totalEpochs=ARG.EPOCH_NUM,
            saveEvery=1,
            batchSize=ARG.BATCH_SIZE,
        )
