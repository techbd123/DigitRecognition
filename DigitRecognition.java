import java.util.*;
import java.io.*;
import javax.imageio.*;
import java.awt.image.*;

class NeuralNetwork
{
	public NeuralNetwork()
	{

	}
	void Perform()
	{

	}
}

class Digit
{
	String label;
	BufferedImage image;
	int height;
	int width;
	int[] pixel;
	public Digit(BufferedImage image)
	{
		this.image=image;
		ProcessPixel(image);
	}
	void ProcessPixel(BufferedImage image)
	{
		if(image==null) return ;
		this.width=image.getWidth();
		this.height=image.getHeight();
		byte[] temp=((DataBufferByte)image.getRaster().getDataBuffer()).getData();
		int[] rgb=new int[temp.length];
		int[] pixel=new int[rgb.length/3];
		for(int i=0,j=0;i<rgb.length;i++)
		{
			rgb[i]=temp[i];
			rgb[i]=(rgb[i]+256)%256;
			if(i%3==2)
			{
				if(2*(rgb[i]+rgb[i-1]+rgb[i-2])>765) pixel[j++]=0;else pixel[j++]=1;
			}
		}
		this.pixel=pixel;
		return ;
	}
	String ShowPixels()
	{
		String string="";
		for(int i=0;i<pixel.length;i++)
		{
			string+=pixel[i];
			if(i%width==width-1) string+="\r\n";
		}
		DigitRecognition.Show("%s",string);
		return string;
	}
}

public class DigitRecognition
{
	static File imageFile=null;
	static BufferedImage image=null;
	static Digit digit=null;
	public static void main(String[] args)
	{
		Initialize(args);
		digit=new Digit(image);
		digit.ShowPixels();
		NeuralNetwork nn=new NeuralNetwork();
		nn.Perform();
	}
	static void Initialize(String[] args)
	{
		imageFile=SetImageFile(args[0]);
		image=SetImage(imageFile);
	}
	public static void Show(Object obj)
	{
		System.out.println(obj);
	}
	public static void Show(String format,Object...arguments)
	{
		System.out.printf(format,arguments);
	}
	static File SetImageFile(String filepath)
	{
		File imgFile=null;
		try
		{
			imgFile=new File(filepath);
		}
		catch(Exception e)
		{
			Show(e);
		}
		return imgFile;
	}
	static BufferedImage SetImage(File image)
	{
		BufferedImage img=null;
		try
		{
			img=ImageIO.read(image);
		}
		catch(Exception e)
		{
			Show(e);
		}
		return img;
	}
}