function readNIFTI(name, data) {
    let startTime=performance.now();
    let niftiHeader, niftiImage;
    // parse nifti
    if (nifti.isCompressed(data)) {
        data = nifti.decompress(data);
    }

    if (nifti.isNIFTI(data)) {
        niftiHeader = nifti.readHeader(data);
        niftiImage = nifti.readImage(niftiHeader, data);
    }
    let endTime=performance.now();
    console.log(`readNIFTI function took ${(endTime-startTime)/1000} seconds.`);
    main(niftiImage, niftiHeader);
    niftiImage=null;
}

function scaling(niftiHeader, array){
    let newArray=new Int32Array(array.length);
    let slope=niftiHeader.scl_slope;
    let intercept=niftiHeader.scl_inter;
    for (let i = 0 ; i< array.length; i++){
        newArray[i]=slope*array[i]+intercept;
    }
    return newArray;
}

function niftiToTensor(arrayBuffer, dim1, dim2, dim3){
    let InputTensor1=tf.tensor3d(arrayBuffer, [dim3, dim1, dim2]);
    let InputTensor2=tf.transpose(InputTensor1, [2,1,0]);
    return InputTensor2;
}

function Orientation(InputTensor, orientation){
    let InputTensor1=InputTensor;

    if (orientation[0]=='L'){
        InputTensor1=InputTensor1.reverse(0);
    }
    if(orientation[1]=='P'){
        InputTensor1=InputTensor1.reverse(1);
    }
    if (orientation[2]=='I'){
        InputTensor1=InputTensor1.reverse(2);
    }
    return InputTensor1;
}

function Padding(dim, patch){
    let iDim=0;
    while(iDim * patch < dim){
        iDim++;
    }
    return iDim;
}

function Normalization(min, max, array){
    let normArr=[];
    for (let index = 0; index < array.length; index++) {
        if (array[index] > max) {
            normArr.push(1.0);
            //count1++;
        } else if (array[index] < min) {
            normArr.push(0);
            //count2++;
        } else {
            normArr.push((array[index] + Math.abs(min)) / (Math.abs(min)+Math.abs(max)));
            // count3++;
        }
    }
    return normArr;
}

function softmax_First_Dim(InputTensor){
    let startTime = performance.now();
    let InputTensor1 = tf.transpose(InputTensor);
    let InputTensor2 = InputTensor1.softmax();
    InputTensor1.dispose();
    let InputTensor3 = tf.transpose(InputTensor2);
    InputTensor2.dispose();
    let endTime = performance.now();
    console.log(`Doing the softmax took ${(endTime - startTime) / 1000} seconds.`);
    return InputTensor3;
}



function niftiParser(niftiHeader, niftiImage){
    let typedData;
    if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
        typedData = new Uint8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
        typedData = new Int16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32) {
        typedData = new Int32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
        typedData = new Float32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) {
        typedData = new Float64Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT8) {
        typedData = new Int8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT16) {
        typedData = new Uint16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT32) {
        typedData = new Uint32Array(niftiImage);
    } else {
        return;
    }
    return typedData;
}

function drawCanvas(plane, canvas, canvas2, canvas_CT, canvas_Label, slice, niftiHeader, outputTensor, input) {
    let startTime=performance.now();
    console.log(slice);
    slice=parseInt(slice);
    // get nifti dimensions
    let Map3D=outputTensor;
    let dim1 = Map3D.shape[0];
    let dim2 = Map3D.shape[1];
    let dim3 = Map3D.shape[2];
    let labelSlice;
    let arrayCT;
    let w,h;
    if(plane=="Sagittal"){
        labelSlice=Map3D.slice([slice,0,0], [1,dim2,dim3]).reshape([dim2,dim3]).arraySync();
        arrayCT=input.slice([slice,0,0], [1,dim2,dim3]).reshape([dim2,dim3]).arraySync();
        w=dim2;
        h=dim3;

    }
    if(plane=="Coronal"){
        labelSlice=Map3D.slice([0,slice,0], [dim1,1,dim3]).reshape([dim1,dim3]).arraySync();
        arrayCT=input.slice([0,slice,0], [dim1,1,dim3]).reshape([dim1,dim3]).arraySync();
        w=dim1;
        h=dim3;
    }
    if(plane=="Axial"){
        labelSlice=Map3D.slice([0,0,slice], [dim1,dim2,1]).reshape([dim1,dim2]).arraySync();
        arrayCT=input.slice([0,0,slice], [dim1,dim2,1]).reshape([dim1,dim2]).arraySync();
        w=dim1;
        h=dim2;

    }
    // set canvas dimensions to nifti slice dimensions
    canvas.width = w;
    canvas.height = h;
    canvas2.width = w;
    canvas2.height = h;
    canvas_CT.width = w;
    canvas_CT.height = h;
    canvas_Label.width = w;
    canvas_Label.height = h;

    // make canvas image data
    let ctx = canvas.getContext("2d");
    let ctx2 = canvas2.getContext("2d");
    let ctx_CT=canvas_CT.getContext("2d");
    let ctx_Label=canvas_Label.getContext("2d");
    let canvasImageData = ctx.createImageData(canvas.width, canvas.height);
    let canvasImageDataL=ctx2.getImageData(0,0,canvas2.width, canvas2.height);
    let canvasImageData_CT = ctx_CT.createImageData(canvas.width, canvas.height);
    let canvasImageData_Label = ctx_Label.createImageData(canvas.width, canvas.height);
    // convert raw data to typed array based on nifti datatype


    let count=0;
    let count1=0;
    let count2=0;
    let count3=0;
    let count4=0;
    let count5=0;
    let end1;
    let end2;
    if (plane=="Sagittal"){
        end1=dim3;
        end2=dim2;
    }
    if(plane=="Coronal"){
        end1=dim3;
        end2=dim1;
    }
    if(plane=="Axial"){
        end1=dim1;
        end2=dim2;
    }

    for (let i2 =0; i2<end1; i2++) {
        for (let j2 = 0; j2 < end2; j2 ++) {
            let value;
            let valueCT;
            if (plane=="Sagittal" || plane=="Coronal"){
                value =labelSlice[j2][i2];
                valueCT=arrayCT[j2][i2];
            }
            else if(plane=="Axial"){
                value =labelSlice[i2][j2];
                valueCT=arrayCT[i2][j2];
            }
            if (valueCT<-1024){
                valueCT=-1024;
            }else if(valueCT>600){
                valueCT=600;
            }
            canvasImageData.data[count*4]=Math.floor(255/(600+1024)*(valueCT+1024));
            canvasImageData.data[count*4+1]=Math.floor(255/(600+1024)*(valueCT+1024));
            canvasImageData.data[count*4+2]=Math.floor(255/(600+1024)*(valueCT+1024));
            canvasImageData.data[count*4+3]=0xFF;
            /*
               Assumes data is 8-bit, otherwise you would need to first convert
               to 0-255 range based on datatype range, data range (iterate through
               data to find), or display range (cal_min/max).

               Other things to take into consideration:
                 - data scale: scl_slope and scl_inter, apply to raw value before
                   applying display range
                 - orientation: displays in raw orientation, see nifti orientation
                   info for how to orient data
                 - assumes voxel shape (pixDims) is isometric, if not, you'll need
                   to apply transform to the canvas
                 - byte order: see littleEndian flag
            */
            if(value<0.5){
                canvasImageDataL.data[count * 4] = 0;
                canvasImageDataL.data[count * 4 + 1] = 0;
                canvasImageDataL.data[count * 4 + 2] = 0;
                canvasImageDataL.data[count * 4 + 3] = 0;
            }else if(value<1.2){
                count1++;
                canvasImageDataL.data[count * 4] = 138;
                canvasImageDataL.data[count * 4 + 1] = 12;
                canvasImageDataL.data[count * 4 + 2] = 81;
                canvasImageDataL.data[count * 4 + 3] = 150;
            }else if(value<2.2){
                count2++;
                canvasImageDataL.data[count * 4] = 167;
                canvasImageDataL.data[count * 4 + 1] = 159;
                canvasImageDataL.data[count * 4 + 2] = 39;
                canvasImageDataL.data[count * 4 + 3] = 150;
            }else if(value<3.2){
                count3++;
                canvasImageDataL.data[count * 4] = 231;
                canvasImageDataL.data[count * 4 + 1] = 241;
                canvasImageDataL.data[count * 4 + 2] = 2;
                canvasImageDataL.data[count * 4 + 3] = 150;
            }else if(value<4.2){
                count4++;
                canvasImageDataL.data[count * 4] = 38;
                canvasImageDataL.data[count * 4 + 1] = 253;
                canvasImageDataL.data[count * 4 + 2] = 244;
                canvasImageDataL.data[count * 4 + 3] = 150;
            }else{
                count5++;
                canvasImageDataL.data[count * 4] = 216;
                canvasImageDataL.data[count * 4 + 1] = 107;
                canvasImageDataL.data[count * 4 + 2] = 211;
                canvasImageDataL.data[count * 4 + 3] = 150;
            }
            count++;
        }
    }

    console.log(`label value=1 : ${count1}`);
    console.log(`label value=2 : ${count2}`);
    console.log(`label value=3 : ${count3}`);
    console.log(`label value=4 : ${count4}`);
    console.log(`label value=5 : ${count5}`);

    ctx.putImageData(canvasImageData, 0, 0);
    ctx_CT.putImageData(canvasImageData, 0, 0);
    ctx2.putImageData(canvasImageDataL,0,0);
    ctx_Label.putImageData(canvasImageDataL,0,0);
    ctx.drawImage(canvas2,0,0);
    let endTime=performance.now();
}

function makeSlice(file, start, length) {
    let fileType = (typeof File);

    if (fileType === 'undefined') {
        return function () {};
    }

    if (File.prototype.slice) {
        return file.slice(start, start + length);
    }

    if (File.prototype.mozSlice) {
        return file.mozSlice(start, length);
    }

    if (File.prototype.webkitSlice) {
        return file.webkitSlice(start, length);
    }

    return null;
}

function readFile(file) {
    let blob = makeSlice(file, 0, file.size);

    let reader = new FileReader();

    reader.onloadend = function (evt) {
        if (evt.target.readyState === FileReader.DONE) {
            readNIFTI(file.name, evt.target.result);
        }
    };

    reader.readAsArrayBuffer(blob);
}

function handleFileSelect(evt) {
    let files = evt.target.files;
    readFile(files[0]);
    document.getElementById("progress").innerHTML='Model running...';
}