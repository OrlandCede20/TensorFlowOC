package com.example.tensorflowoc

import androidx.appcompat.app.AppCompatActivity
import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.example.tensorflowoc.ml.Cantantes
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {
    private lateinit var result: TextView
    private lateinit var confidence: TextView
    private lateinit var imageView: ImageView
    private lateinit var picture: Button
    private lateinit var linearlay:LinearLayout
    private lateinit var linearlay2:LinearLayout
    private val imageSize = 224

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        result = findViewById(R.id.result)
        confidence = findViewById(R.id.confidence)
        imageView = findViewById(R.id.imageView)
        linearlay=findViewById(R.id.rel1)
        linearlay2=findViewById(R.id.rel2)



        linearlay.setOnClickListener {

            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 1)
            } else {

                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        linearlay2.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, 2)
        }
    }

    private fun classifyImage(image: Bitmap) {
        try {
            val model = Cantantes.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val `val` = intValues[pixel++] // RGB
                    byteBuffer.putFloat(((`val` shr 16) and 0xFF) * (1.0f / 255.0f))
                    byteBuffer.putFloat(((`val` shr 8) and 0xFF) * (1.0f / 255.0f))
                    byteBuffer.putFloat((`val` and 0xFF) * (1.0f / 255.0f))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.getOutputFeature0AsTensorBuffer()

            val confidences = outputFeature0.floatArray
            var maxPos = 0
            var maxConfidence = 0.0f
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    maxPos = i
                }
            }

            val classes = arrayOf("ARCANGEL", "DADDY YANKEE", "TITO EL BAMBINO", "WISIN")

            result.text = classes[maxPos]

            var s = ""
            for (i in classes.indices) {
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100)
            }
            confidence.text = s
            model.close()
        } catch (e: IOException) {

        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == 1 && resultCode == RESULT_OK) {

            val image = data?.extras?.get("data") as Bitmap
            val dimension = Math.min(image.width, image.height)
            val thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
            imageView.setImageBitmap(thumbnail)
            val scaledImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
            classifyImage(scaledImage)
        } else if (requestCode == 2 && resultCode == RESULT_OK && data != null) {

            val selectedImage = data.data
            val imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImage)
            val dimension = Math.min(imageBitmap.width, imageBitmap.height)
            val thumbnail = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension)
            imageView.setImageBitmap(thumbnail)
            val scaledImage = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false)
            classifyImage(scaledImage)
        }
        super.onActivityResult(requestCode, resultCode, data)
    }

}
