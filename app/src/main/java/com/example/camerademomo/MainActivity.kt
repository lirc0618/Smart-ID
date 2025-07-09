package com.example.camerademomo

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.content.FileProvider
import java.io.File

import com.example.camerademomo.Pufname

import org.pytorch.IValue
import org.pytorch.MemoryFormat
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.exp
import kotlin.math.round

class MainActivity : ComponentActivity() {

    private lateinit var imageView: ImageView
    var takePhoto = 1
    var fromAlbum = 2
    var bitmapout: Bitmap? = null
    lateinit var imageUri: Uri
    lateinit var outputImage: File
    var module_01: Module? = null

    // 模型归一化参数（与训练阶段一致）
    private val MODEL_MEAN = floatArrayOf(0.5461f, 0.5460f, 0.5675f)
    private val MODEL_STD = floatArrayOf(0.1151f, 0.1151f, 0.0822f)
    private val INPUT_SIZE = 224

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
//            module_01 = Module.load(assetFilePath(this, "model.pt"))
//            module_01 = Module.load(assetFilePath(this, "model_1.pt"))
            module_01 = Module.load(assetFilePath(this, "model_0411_2_scripted.pt"))
            Log.d("PytorchHelloWorld", "Successful4")
        } catch (e: IOException) {
            Log.e("PytorchHelloWorld", "Error reading assets", e)
            finish()
        }

        val takephotobutton = findViewById<Button>(R.id.button)
        imageView = findViewById(R.id.imageView)

        takephotobutton.setOnClickListener {
            // 创建File对象，用于存储拍照后的图片
            // 调用 getExternalCacheDir() 获取应用关联缓存目录
            outputImage = File(externalCacheDir, "output_image.jpg")
            //如果存在，先删除
            if (outputImage.exists()) {
                outputImage.delete()
            }
            outputImage.createNewFile()
            // 将 File 对象转换为 Uri 对象，Uri 对象标识图片的本地真实路径
            // 如果系统版本低于 Android 7.0，就调用 Uri 的 fromFile() 方法
            // 否则调用 FileProvider.getUriForFile() 方法，该方法接收三个参数：
            // 参数1：Context 对象
            // 参数2：任意唯一的字符串
            // 参数3：刚刚创建的 File 对象
            // 选择性的将封装的Uri分享给外部
            imageUri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                FileProvider.getUriForFile(
                    this,
                    "com.example.camerademomo.fileprovider",
                    outputImage
                )
            } else {
                Uri.fromFile(outputImage)
            }
            // 启动相机程序
            val intent = Intent("android.media.action.IMAGE_CAPTURE")
            intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri)
            startActivityForResult(intent, takePhoto)
        }

        val fromAlbumBt = findViewById<Button>(R.id.inputbutton)
        fromAlbumBt.setOnClickListener {
            //打开系统的文件选择器
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            //过滤 只显示图片
            intent.type = "image/*"
            startActivityForResult(intent, fromAlbum)
        }

        val runModelButton = findViewById<Button>(R.id.runModelButton)
        runModelButton.setOnClickListener {
//            // 确保 bitmapout 不为空
            bitmapout?.let {

//                val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
//                    it,
//                    floatArrayOf(0.5461f, 0.5460f, 0.5675f),
//                    floatArrayOf(0.1151f, 0.1151f, 0.0822f),
//                    MemoryFormat.CONTIGUOUS
//                ) // 维度：[1,3,224,224]
//
//                // 执行推理
//                val startTime_01 = System.currentTimeMillis()
//                val outputTensor_01: Tensor? = module_01?.forward(IValue.from(inputTensor))?.toTensor()
//                val endTime_01 = System.currentTimeMillis()
//                val InferenceTime01 = endTime_01 - startTime_01
//
//                outputTensor_01?.let { tensor ->
//                    val logits = tensor.dataAsFloatArray
//                    val maxLogit = logits.maxOrNull() ?: Float.NEGATIVE_INFINITY
//                    val zExp = logits.map { exp(it - maxLogit) }
//                    val sumZExp = zExp.sum()
//                    val softmax = zExp.map { round(it / sumZExp * 100000.0) / 100000.0 }
//
//                    Log.d("PytorchHelloWorld", "Softmax: $softmax")
//
//                    var maxScore = -Float.MAX_VALUE
//                    var maxScoreIdx = -1
//                    for (i in softmax.indices) {
//                        if (softmax[i] > maxScore) {
//                            maxScore = softmax[i].toFloat()
//                            maxScoreIdx = i
//                        }
//                    }
//
//                    val className: String = Pufname.PUF_CLASS[maxScoreIdx]
//
//                    val textView: TextView = findViewById(R.id.textout)
//                    val tex = """
//                    Inference Result：$className
//                    Inference Time：${InferenceTime01}ms
//                    """.trimIndent()
//                    textView.text = tex
//                }
//            } ?: run {
//                Log.e("PytorchHelloWorld", "No image available for prediction")
//            }


//            val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
//                it,
//                floatArrayOf(0.5461f, 0.5460f, 0.5675f),
//                floatArrayOf(0.1151f, 0.1151f, 0.0822f),
//                MemoryFormat.CONTIGUOUS
//            )

                try {
                    // 手动转换Bitmap为Tensor（CHW格式）
                    val inputTensor = convertBitmapToTensor(it)
//                    inputTensor.memoryFormat() = CONTIGUOUS

                    // 执行推理
                    val startTime = System.currentTimeMillis()
                    val outputTensor = module_01?.forward(IValue.from(inputTensor))?.toTensor()
                    val endTime = System.currentTimeMillis()

                    outputTensor?.let { tensor ->
                        val scores = tensor.dataAsFloatArray
                        val maxScoreIdx = scores.indices.maxByOrNull { scores[it] } ?: -1

                        if (maxScoreIdx != -1) {
                            val className = Pufname.PUF_CLASS[maxScoreIdx]
                            val confidence = softmax(scores)[maxScoreIdx]

                            val textView: TextView = findViewById(R.id.textout)
//                            置信度：${String.format("%.2f%%", confidence * 100)}
                            val resultText = """
                                Inference Result：$className
                                Inference Time：${endTime - startTime}ms
                            """.trimIndent()
                            textView.text = resultText
                        }
                    }
                } catch (e: Exception) {
                    Log.e("PytorchHelloWorld", "Inference error: ${e.message}")
                }
            } ?: run {
                Log.e("PytorchHelloWorld", "No image available for prediction")
            }
        }
    }

    private fun convertBitmapToTensor(bitmap: Bitmap): Tensor {
        // 确保图片尺寸为224x224
        val resizedBitmap = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else {
            bitmap
        }

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // 创建符合PyTorch要求的CHW格式数组 [3, 224, 224]
        val floatValues = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = pixels[y * INPUT_SIZE + x]

                // 提取RGB通道并归一化（注意顺序：R->0, G->1, B->2）
                floatValues[0 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                    ((pixel shr 16) and 0xFF) / 255.0f // R
                floatValues[1 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                    ((pixel shr 8) and 0xFF) / 255.0f  // G
                floatValues[2 * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                    (pixel and 0xFF) / 255.0f          // B
            }
        }

        // 应用归一化参数
        for (c in 0 until 3) {
            for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
                val idx = c * INPUT_SIZE * INPUT_SIZE + i
                floatValues[idx] = (floatValues[idx] - MODEL_MEAN[c]) / MODEL_STD[c]
            }
        }

        // 创建Tensor，注意维度顺序：[1, 3, 224, 224]
        return Tensor.fromBlob(floatValues, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    // 计算Softmax概率
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = FloatArray(logits.size)
        var sum = 0.0f

        for (i in logits.indices) {
            expValues[i] = exp(logits[i] - maxLogit)
            sum += expValues[i]
        }

        for (i in expValues.indices) {
            expValues[i] /= sum
        }

        return expValues
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            takePhoto -> {
                if (resultCode == RESULT_OK) {
                    val bitmap = decodeSampledBitmapFromUri(imageUri)
                    bitmapout = bitmap
                    imageView.setImageBitmap(bitmap?.let { rotateIfRequired(it) })
                }
            }

            fromAlbum -> {
                if (resultCode == RESULT_OK && data != null) {
                    data.data?.let { uri ->
                        val bitmap = decodeSampledBitmapFromUri(uri)
                        bitmapout = bitmap
                        imageView.setImageBitmap(bitmap)
                    }
                }
            }
        }
    }

    private fun decodeSampledBitmapFromUri(uri: Uri): Bitmap? {
        try {
            val options = BitmapFactory.Options()
            options.inJustDecodeBounds = true
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }

            options.inSampleSize = calculateInSampleSize(options, INPUT_SIZE, INPUT_SIZE)
            options.inJustDecodeBounds = false

            return contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }
        } catch (e: Exception) {
            Log.e("ImageUtils", "Error decoding bitmap: ${e.message}")
            return null
        }
    }

//    private fun decodeSampledBitmapFromUri(uri: Uri): Bitmap? {
//        try {
//            // 先获取原始图片尺寸
//            val options = BitmapFactory.Options()
//            options.inJustDecodeBounds = true
//            contentResolver.openInputStream(uri)?.use { stream ->
//                BitmapFactory.decodeStream(stream, null, options)
//            }
//
//            // 计算采样率，避免加载过大图片到内存
//            options.inSampleSize = calculateInSampleSize(options, 224, 224)
//            options.inJustDecodeBounds = false
//
//            // 加载缩放后的图片
//            val bitmap = contentResolver.openInputStream(uri)?.use { stream ->
//                BitmapFactory.decodeStream(stream, null, options)
//            } ?: return null
//
//            // 强制缩放并居中裁剪到224x224
//            return centerCropAndResize(bitmap, 224, 224)
//        } catch (e: Exception) {
//            Log.e("ImageUtils", "Error decoding bitmap: ${e.message}")
//            return null
//        }
//    }

    private fun centerCropAndResize(bitmap: Bitmap, reqWidth: Int, reqHeight: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // 计算裁剪区域
        val minDimension = minOf(width, height)
        val left = (width - minDimension) / 2
        val top = (height - minDimension) / 2

        // 居中裁剪
        val croppedBitmap = Bitmap.createBitmap(
            bitmap, left, top, minDimension, minDimension
        )

        // 缩放至目标尺寸
        return Bitmap.createScaledBitmap(croppedBitmap, reqWidth, reqHeight, true)

//        // 先缩放至256x256（模拟训练阶段的RandomResizedCrop范围）
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
//        // 中心裁剪至224x224
//        val width = scaledBitmap.width
//        val height = scaledBitmap.height
//        val left = (width - 224) / 2
//        val top = (height - 224) / 2
//        return Bitmap.createBitmap(scaledBitmap, left, top, 224, 224)
    }
//
//    private fun decodeSampledBitmapFromUri(uri: Uri, reqWidth: Int, reqHeight: Int): Bitmap? {
//        val options = BitmapFactory.Options()
//        options.inJustDecodeBounds = true
//        contentResolver.openInputStream(uri).use { stream ->
//            BitmapFactory.decodeStream(stream, null, options)
//        }
//
//        options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight)
//        options.inJustDecodeBounds = false
//        return contentResolver.openInputStream(uri)?.use { stream ->
//            BitmapFactory.decodeStream(stream, null, options)
//        }
//    }
//
    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val height = options.outHeight
        val width = options.outWidth
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight = height / 2
            val halfWidth = width / 2

            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    private fun getBitmapFromUri(uri: Uri) = contentResolver.openFileDescriptor(uri, "r")?.use {
        BitmapFactory.decodeFileDescriptor(it.fileDescriptor)
    }

    //  调用原相机可能会在一些手机发生照片旋转
    private fun rotateIfRequired(bitmap: Bitmap): Bitmap {
        val exif = ExifInterface(outputImage.path)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270)
            else -> bitmap
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degree: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degree.toFloat())
        val rotatedBitmap =
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        // 将不需要的Bitmap对象回收
        bitmap.recycle()
        return rotatedBitmap
    }

    companion object {
        /**
         * Copies specified asset to the file in /files app directory and returns this file absolute path.
         *
         * @return 文件绝对路径
         */
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String?): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            context.assets.open(assetName!!).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while ((`is`.read(buffer).also { read = it }) != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }
    }
}
