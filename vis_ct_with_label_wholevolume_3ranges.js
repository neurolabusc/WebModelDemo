
// use an async context to call onnxruntime functions.
async function main(niftiImage, niftiHeader) {
    let time=performance.now();
    let startTime = performance.now();
    await tf.setBackend('wasm');
    let endTime=performance.now();
    console.log(`Setting backend took ${(endTime-startTime)/1000} seconds.`);
    console.log(tf.getBackend());
    startTime = performance.now();
    let dim1 = niftiHeader.dims[1];
    let dim2 = niftiHeader.dims[2];
    let dim3=niftiHeader.dims[3];
    // convert raw data to typed array based on nifti datatype
    let typedData=niftiParser(niftiHeader, niftiImage);
    let typedData2 = scaling(niftiHeader, typedData);
    endTime=performance.now();
    console.log(`Converting the input array took ${(endTime-startTime)/1000} seconds.`);
    startTime = performance.now();
    const InputTensor2=niftiToTensor(typedData2, dim1, dim2, dim3);
    typedData=null;
    typedData2=null;
    endTime=performance.now();
    console.log(`Creating the input tensor took ${(endTime-startTime)/1000} seconds.`);
    console.log(tf.getBackend());
    console.log(`the shape of the volume is: (should be 512*512*height) ${InputTensor2.shape}`);
    startTime = performance.now();
    // let InputTensor4=InputTensor2.reverse(1);
    let InputTensor3=await Orientation(InputTensor2, "LAS");
    InputTensor2.dispose();
    //InputTensor4.dispose();
    endTime=performance.now();
    console.log(`reverse the dimension took ${(endTime-startTime)/1000} seconds.`);

    startTime=performance.now();

    let multiplier_dim1=Padding(dim1, 96);
    let multiplier_dim2=Padding(dim2, 96);
    let multiplier_dim3=Padding(dim3, 96);

    console.log(`multiplier of dim3 = ${multiplier_dim3}`);
    const paddingsize_dim1=96*multiplier_dim1;
    const paddingsize_dim2=96*multiplier_dim2;
    const paddingsize_dim3=96*multiplier_dim3;
    let newBuffer=tf.buffer([paddingsize_dim1, paddingsize_dim2, paddingsize_dim3]);
    endTime=performance.now();
    console.log(`creating the buffer of padding took ${(endTime-startTime)/1000} seconds.`);
    startTime=performance.now();
    let temp = InputTensor3.bufferSync();
    endTime=performance.now();
    console.log(`creating the buffer of the original data took ${(endTime-startTime)/1000} seconds.`);
    startTime=performance.now();
    console.log(tf.getBackend());
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
    temp=null;
    endTime=performance.now();
    console.log(`loop of padding took ${(endTime-startTime)/1000} seconds.`);
    console.log(newBuffer);
    startTime=performance.now();
    let newTensor=newBuffer.toTensor();
    InputTensor3.dispose();
    console.log(`the shape of the tensor after padding is ${newTensor.shape}`);
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
    let outputBuffer=tf.buffer([paddingsize_dim1, paddingsize_dim2, paddingsize_dim3], 'bool');

    session = await ort.InferenceSession.create('./Unet512.onnx', {executionProviders: ['wasm']});

    endTime=performance.now();
    console.log(`creating the inference session took ${(endTime-startTime)/1000} seconds.`);
    for (let i=0; i<multiplier_dim1; i++) {
        for (let j = 0; j < multiplier_dim2; j++) {
            for (let k = 0; k < multiplier_dim3; k++) {
                startTime = performance.now();
                patchTensor = newTensor.slice([i*96, j*96, k*96], [96, 96, 96]);
                patch = patchTensor.dataSync();
                patchTensor.dispose();
                norm = Normalization(-1024, 600, patch);
                thePatch = new Float32Array(norm);
                norm=null;
                patch=null;
                endTime = performance.now();
                console.log(`getting the patch#${cntPatch} took ${(endTime - startTime) / 1000} seconds.`);
                timePatch=timePatch+(endTime-startTime)/1000;
                startTime = performance.now();
                console.log(`the size of the patch (should be 884736) is ${thePatch.length}`);
                tensor = new ort.Tensor('float32', thePatch, [1, 1, 96, 96, 96]);
                feeds = {input: tensor};
                output = await session.run(feeds);
                endTime = performance.now();
                console.log(`running the model took ${(endTime - startTime) / 1000} seconds.`);
                timeModelRunning=timeModelRunning+(endTime-startTime)/1000;
                startTime = performance.now();
                outputData = output.output.data;
                thePatch=null;
                tensor=null;
                arr = new Float32Array(outputData);
                outputData=null;
                output=null;
                endTime = performance.now();
                console.log(`converting the typed array took ${(endTime - startTime) / 1000} seconds.`);
                startTime = performance.now();
                b = tf.tensor4d(arr, [6, 96, 96, 96]);
                arr=null;
                endTime = performance.now();
                console.log(`Creating the 4d tensor took ${(endTime - startTime) / 1000} seconds.`);
                d=softmax_First_Dim(b);
                b.dispose();
                startTime = performance.now();
                d_reshape=d.reshape([6,96,96,96]);
                d.dispose();
                startTime = performance.now();
                d_argMax=d_reshape.argMax(0).reshape([96,96,96]);
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

                d_arr=null;
                endTime = performance.now();
                console.log(`storing the output patch#${cntPatch} took ${(endTime - startTime) / 1000} seconds.`);
                cntPatch++;
            }
        }
    }
    startTime=performance.now();
    AM=outputBuffer.toTensor();
    outputBuffer=null;
    endTime=performance.now();
    console.log(`calling the toTensor of the whole buffer took ${(endTime - startTime) / 1000} seconds.`);
    console.log(`making patches took (in total)${timePatch} seconds. Average: ${timePatch/cntPatch}`);
    console.log(`running the model for the whole volume took ${timeModelRunning} seconds. Average: ${timeModelRunning/cntPatch}`);
    startTime=performance.now();
    const outputTensor=AM.slice([0,0,0], [dim1, dim2, dim3]);
    endTime=performance.now();
    console.log(`Taking the slice to remove padding took ${(endTime - startTime) / 1000} seconds.`);
    AM.dispose();
    let ttime=performance.now();
    console.log(`the whole main function took ${(ttime-time)/1000} seconds`);
    console.log(`the pre/post processing took ${(ttime-time)/1000-timeModelRunning} seconds`);


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

    let slider1 = document.getElementById('myRange_1');
    let slider2=document.getElementById('myRange_2');
    let slider3=document.getElementById('myRange_3');
    let slices1 = niftiHeader.dims[1];
    slider1.max = slices1 - 1;
    slider1.value = 140;
    slider1.oninput = function() {
        drawCanvas("Sagittal", canvas1, canvas21, canvas_CT1, canvas_Label1, slider1.value, niftiHeader, outputTensor, newTensor);
    };
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
    document.getElementById("progress").innerHTML=' ';
}
