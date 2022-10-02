
// use an async context to call onnxruntime functions.
async function main(niftiImage, niftiHeader) {
    let time=performance.now();
    //performance.now() is used to record the running time of each part
    let startTime = performance.now();
    
    await tf.setBackend('wasm');
    //set the backend executer of TensorFlow operations to WebAssembly
    
    let endTime=performance.now();
    console.log(`Setting backend took ${(endTime-startTime)/1000} seconds.`);
    console.log(tf.getBackend());
    
    startTime = performance.now();
    let dim1 = niftiHeader.dims[1];
    let dim2 = niftiHeader.dims[2];
    let dim3=niftiHeader.dims[3];
    //get the image dimensions from the nifti file header
    
    let typedData=niftiParser(niftiHeader, niftiImage);
    // convert raw data to JS typed array based on nifti datatype
    
    let typedData2 = scaling(niftiHeader, typedData);
    // apply the scaling specified on nifti file header on the original data to get the real data of the image
    
    endTime=performance.now();
    console.log(`Converting the input array took ${(endTime-startTime)/1000} seconds.`);
    
    startTime = performance.now();
    const InputTensor2=niftiToTensor(typedData2, dim1, dim2, dim3);
    //convert the JS typed array of the image file to a TensorFlow tensor
    
    typedData=null;
    typedData2=null;
    endTime=performance.now();
    console.log(`Creating the input tensor took ${(endTime-startTime)/1000} seconds.`);
    
    console.log(tf.getBackend());
    //check if the backend executer of TensorFlow is changed successfully
    
    console.log(`the shape of the volume is: (should be 512*512*height) ${InputTensor2.shape}`);
    
    startTime = performance.now();
    let InputTensor3=await Orientation(InputTensor2, "LAS");
    //apply the orienting to the image tensor
    
    InputTensor2.dispose();
    //manually dispose the tensor objects that are no longer used to conduct garbage collection
    
    endTime=performance.now();
    console.log(`reverse the dimension took ${(endTime-startTime)/1000} seconds.`);

    startTime=performance.now();

    let multiplier_dim1=Padding(dim1, 96);
    let multiplier_dim2=Padding(dim2, 96);
    let multiplier_dim3=Padding(dim3, 96);
    //compute n where n is the smallest number such that n*96>each dimension, 
    //so we know how many patches can be cropped after the padding
    
    console.log(`multiplier of dim3 = ${multiplier_dim3}`);
    const paddingsize_dim1=96*multiplier_dim1;
    const paddingsize_dim2=96*multiplier_dim2;
    const paddingsize_dim3=96*multiplier_dim3;
    //compute the dimensions of padded tensor
    
    let newBuffer=tf.buffer([paddingsize_dim1, paddingsize_dim2, paddingsize_dim3]);
    //creating a new TensorFlow buffer object that stores the padded tensor values.
    endTime=performance.now();
    console.log(`creating the buffer of padding took ${(endTime-startTime)/1000} seconds.`);
    
    startTime=performance.now();
    let temp = InputTensor3.bufferSync();
    //get the buffer of the current tensor for the iteration later.
    endTime=performance.now();
    console.log(`creating the buffer of the original data took ${(endTime-startTime)/1000} seconds.`);
    
    startTime=performance.now();
    console.log(tf.getBackend());
    //check the backend executer again
    
    console.log(temp);
    for (let i=0; i<paddingsize_dim1; i++){
        for (let j=0; j<paddingsize_dim2; j++){
            for (let k=0; k<paddingsize_dim3; k++){
                if (i<InputTensor3.shape[0]
                    && j<InputTensor3.shape[1]
                    && k<InputTensor3.shape[2]){
                    newBuffer.set(temp.get(i,j,k), i,j,k);
                }else{
                    newBuffer.set(-1024,i,j,k);
                }
            }
        }
    }
    //implementation of padding
    //loop through the buffer of image tensor, assign the value inside it into the new buffer, and pad the area outside it with HU unit of -1024 
    
    temp=null;
    endTime=performance.now();
    console.log(`loop of padding took ${(endTime-startTime)/1000} seconds.`);
    console.log(newBuffer);
    
    startTime=performance.now();
    let newTensor=newBuffer.toTensor();
    //convert the padded buffer object to TensorFlow tensor
    
    InputTensor3.dispose();
    //it will be no longer used, so dispose it
    
    console.log(`the shape of the tensor after padding is ${newTensor.shape}`);
    //check if the new tensor has the correct shape (dimensions);
    
    endTime=performance.now();
    console.log(`converting the padding buffer to tensor took ${(endTime-startTime)/1000} seconds.`);
    
    let norm=[];
    let patchTensor, patch,
        thePatch, session,
        tensor,
        feeds, output,
        outputData,
        AM, b, iCur, jCur, kCur, intArr,
        d, d_arr, d_reshape, d_argMax;
    let arr;
    let timePatch=0;
    let cntPatch=1;
    let timeModelRunning=0;
    //a series of variables defined for the deployment of the model later. 
    //Some of them might not be used as the version of this program is updating
    
    let outputBuffer=tf.buffer([paddingsize_dim1, paddingsize_dim2, paddingsize_dim3], 'bool');
    //create the output label map buffer to store the result of model deployment.
    
    session = await ort.InferenceSession.create('./Unet512.onnx', {executionProviders: ['wasm']});
    //create the ONNX model inference session and specify the backend executer to be Web Assembly
    
    endTime=performance.now();
    console.log(`creating the inference session took ${(endTime-startTime)/1000} seconds.`);
    
    //the following loop implements the sliding window inference
    //each 96^3 patches of the input tensor is fetched and fed into the model
    //some pre-processing and post-processing transforms are implemented and applied
    for (let i=0; i<multiplier_dim1; i++) {
        for (let j = 0; j < multiplier_dim2; j++) {
            for (let k = 0; k < multiplier_dim3; k++) {
                startTime = performance.now();
                
                patchTensor = newTensor.slice([i*96, j*96, k*96], [96, 96, 96]);
                //fetch each patches
                
                patch = patchTensor.dataSync();
                //fetch the data array inside the patch
                
                patchTensor.dispose();
                //dispose the patchTensor object since we no longer use it in the current iteration step
                
                norm = Normalization(-1024, 600, patch);
                //apply intensity window scaling on the patch data
                //the values will be converted into [0,1]
                
                thePatch = new Float32Array(norm);
                //change the value type to make the type compatible later.
                
                norm=null;
                patch=null;
                //reassign the variable to manually dispose their memory usage
                //it is valid according to my previous experiementation, 
                //but haven't checked documentations or publications that has validated this vigorously
                
                endTime = performance.now();
                console.log(`getting the patch#${cntPatch} took ${(endTime - startTime) / 1000} seconds.`);
                
                timePatch=timePatch+(endTime-startTime)/1000;
                //record the accumulative running time of fetching the patches
                
                startTime = performance.now();
                console.log(`the size of the patch (should be 884736) is ${thePatch.length}`);
                tensor = new ort.Tensor('float32', thePatch, [1, 1, 96, 96, 96]);
                //create the ONNX Runtime(ORT) tensor, which is compatible to ONNX inference session
                
                feeds = {input: tensor};
                output = await session.run(feeds);
                endTime = performance.now();
                console.log(`running the model took ${(endTime - startTime) / 1000} seconds.`);
                //waiting for the model to run
                
                timeModelRunning=timeModelRunning+(endTime-startTime)/1000;
                //record the accumulative running time of model inferencing
                
                startTime = performance.now();
                outputData = output.output.data;
                thePatch=null;
                tensor=null;
                arr = new Float32Array(outputData);
                //store the output data into a JS float32 typed array
                outputData=null;
                output=null;
                //dispose the object
                endTime = performance.now();
                console.log(`converting the typed array took ${(endTime - startTime) / 1000} seconds.`);
                
                startTime = performance.now();
                b = tf.tensor4d(arr, [6, 96, 96, 96]);
                //store the output data into a Tensorflow tensor
                
                arr=null;
                endTime = performance.now();
                console.log(`Creating the 4d tensor took ${(endTime - startTime) / 1000} seconds.`);
                
                d=softmax_First_Dim(b);
                //apply softmax on the tensor
                b.dispose();
                d_reshape=d.reshape([6,96,96,96]);
                //reshape the result tensor of softmax
                d.dispose();
                
                startTime = performance.now();
                d_argMax=d_reshape.argMax(0).reshape([96,96,96]);
                //apply argMax on the tensor
                d_reshape.dispose();
                endTime = performance.now();
                console.log(`doing the argMax took ${(endTime - startTime) / 1000} seconds.`);
                
                startTime=performance.now();
                d_arr=d_argMax.bufferSync();
                d_argMax.dispose();
                iCur=0;
                jCur=0;
                kCur=0;
                let value;
                for(let i5 = i*96; i5<i*96+96 ; i5++) {
                    jCur = 0;
                    for (let j5 = j * 96; j5 < j * 96 + 96; j5++) {
                        kCur = 0;
                        for (let k5 = k * 96; k5 < k * 96 + 96; k5++) {
                            value=d_arr.get( iCur, jCur, kCur);
                            intArr=new Uint8Array([value]);
                            outputBuffer.set(intArr[0], i5, j5, k5);
                            value=null;
                            intArr=null;
                            kCur++;
                        }
                        jCur++;
                    }
                    iCur++;
                }
                //store the result of each input patch tensor into the output buffer according to its original position

                d_arr=null;
                endTime = performance.now();
                console.log(`storing the output patch#${cntPatch} took ${(endTime - startTime) / 1000} seconds.`);
                cntPatch++;
            }
        }
    }
    startTime=performance.now();
    AM=outputBuffer.toTensor();
    //convert the output buffer to tensor
    outputBuffer=null;
    endTime=performance.now();
    console.log(`calling the toTensor of the whole buffer took ${(endTime - startTime) / 1000} seconds.`);
    
    console.log(`making patches took (in total)${timePatch} seconds. Average: ${timePatch/cntPatch}`);
    //log the total patch making time
    
    console.log(`running the model for the whole volume took ${timeModelRunning} seconds. Average: ${timeModelRunning/cntPatch}`);
    //log the total model running time
    
    startTime=performance.now();
    const outputTensor=AM.slice([0,0,0], [dim1, dim2, dim3]);
    endTime=performance.now();
    console.log(`Taking the slice to remove padding took ${(endTime - startTime) / 1000} seconds.`);
    //use slice to remove the padding part of the output tensor
    
    AM.dispose();
    let ttime=performance.now();
    console.log(`the whole main function took ${(ttime-time)/1000} seconds`);
    console.log(`the pre/post processing took ${(ttime-time)/1000-timeModelRunning} seconds`);
    //log the total running time


    let canvas1 = document.getElementById('myCanvas1');
    let canvas_CT1 = document.getElementById('myCanvas_CT1');
    let canvas_Label1 = document.getElementById('myCanvas_Label1');
    let canvas21=document.createElement('canvas');
    let canvas2 = document.getElementById('myCanvas2');
    let canvas_CT2 = document.getElementById('myCanvas_CT2');
    let canvas_Label2 = document.getElementById('myCanvas_Label2');
    let canvas22=document.createElement('canvas');
    let canvas3 = document.getElementById('myCanvas3');
    let canvas_CT3 = document.getElementById('myCanvas_CT3');
    let canvas_Label3 = document.getElementById('myCanvas_Label3');
    let canvas23=document.createElement('canvas');
    //a series of HTML canvas elements for image/label map visualization
    
    let slider1 = document.getElementById('myRange_1');
    let slider2=document.getElementById('myRange_2');
    let slider3=document.getElementById('myRange_3');
    //a series of HTML slider elements, which implements a selection bar to display different slices of images/label maps
    
    let slices1 = niftiHeader.dims[1];
    slider1.max = slices1 - 1;
    slider1.value = 140;
    slider1.oninput = function() {
        drawCanvas("Sagittal", canvas1, canvas21, canvas_CT1, canvas_Label1, slider1.value, niftiHeader, outputTensor, newTensor);
    };
    //the input event handler function that displays different slices when the selections bar is dragged and moved
    
    let slices2 = niftiHeader.dims[2];
    slider2.max = slices2 - 1;
    slider2.value = 140;
    slider2.oninput = function() {
        drawCanvas("Coronal", canvas2, canvas22, canvas_CT2, canvas_Label2, slider2.value, niftiHeader, outputTensor, newTensor);
    };
    
    let slices3 = niftiHeader.dims[3];
    slider3.max = slices3 - 1;
    slider3.value = 140;
    slider3.oninput = function() {
        drawCanvas("Axial", canvas3, canvas23, canvas_CT3, canvas_Label3, slider3.value, niftiHeader, outputTensor, newTensor);
    };
    
    drawCanvas("Sagittal", canvas1, canvas21, canvas_CT1, canvas_Label1, slider1.value, niftiHeader, outputTensor, newTensor);
    drawCanvas("Coronal", canvas2, canvas22, canvas_CT2, canvas_Label2, slider2.value, niftiHeader, outputTensor, newTensor);
    drawCanvas("Axial", canvas3, canvas23, canvas_CT3, canvas_Label3, slider3.value, niftiHeader, outputTensor, newTensor);
    //display the images and label maps when the page is loaded for the first time
    
    document.getElementById("progress").innerHTML=' ';
}
