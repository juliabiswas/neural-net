
import java.io.*;
import java.util.*;

/**
 * Converts BMP files (images) to pel values and converts pel values to BMP files (images).
 *
 * @author EricN
 *         Julia Biswas
 *
 * @version December 3, 2019
 */

/*
 * A member-variable-only class for holding the RGBQUAD C structure elements.
 */
final class RgbQuad
{
   int red;
   int green;
   int blue;
   int reserved; //currently isn't used
}

public class DibDump
{
   //BITMAPFILEHEADER
   static int bmpFileHeader_bfType;          // WORD
   static int bmpFileHeader_bfSize;          // DWORD
   static int bmpFileHeader_bfReserved1;     // WORD
   static int bmpFileHeader_bfReserved2;     // WORD
   static int bmpFileHeader_bfOffBits;       // DWORD

   //BITMAPINFOHEADER
   static int bmpInfoHeader_biSize;          // DWORD
   static int bmpInfoHeader_biWidth;         // LONG
   static int bmpInfoHeader_biHeight;        // LONG
   static int bmpInfoHeader_biPlanes;        // WORD
   static int bmpInfoHeader_biBitCount;      // WORD
   static int bmpInfoHeader_biCompression;   // DWORD
   static int bmpInfoHeader_biSizeImage;     // DWORD
   static int bmpInfoHeader_biXPelsPerMeter; // LONG
   static int bmpInfoHeader_biYPelsPerMeter; // LONG
   static int bmpInfoHeader_biClrUsed;       // DWORD
   static int bmpInfoHeader_biClrImportant;  // DWORD

   static int[][] imageArray; //true color pels

   static boolean topDownDIB = false; //true if the bmpInfoHeader_biHeight is negative; when image is written out,
                                      //it will have a positive bmpInfoHeader_biHeight value

   /*
    * Switches between little and big endian integer formats for integers.
    *
    * @param v       the value (an integer) to do the switching for
    */
   public int swapInt(int v)
   {
      return (v >>> 24) | (v << 24) | ((v << 8) & 0x00FF0000) | ((v >> 8) & 0x0000FF00);
   }

   /*
    * Switches between little and big endian integer formats for shorts.
    *
    * @param v       the value (a short) to do the switching for
    */
   public int swapShort(int v)
   {
      return ((v << 8) & 0xFF00) | ((v >> 8) & 0x00FF);
   }

   /*
    * Accepts an intenger (32 bit) picture element and returns the red, green and blue colors as an RgbQuad object.
    * rgbToPel(int red, int green, int blue) goes the other way.
    *
    * @param pel      the picture element
    *
    * @return         RgbQuad object that contains the r,g,b colors
    */
   public RgbQuad pelToRGB(int pel)
   {
      RgbQuad rgb = new RgbQuad();

      rgb.reserved = 0;

      rgb.blue = pel & 0x00FF;
      rgb.green = (pel >> 8) & 0x00FF;
      rgb.red = (pel >> 16) & 0x00FF;

      return rgb;
   }

   /*
    * Takes red, green and blue color values and returns a single 32-bit integer color.
    * pelToRGB(int pel) for going the other way.
    *
    * @param red        the red color value
    * @param green      the green color value
    * @param blue       the blue color value
    *
    * @return           one 32-bit integer color
    */
   public int rgbToPel(int red, int green, int blue)
   {
      return (red << 16) | (green << 8) | blue;
   }

   /*
    * Accepts an integer (32 bit) picture element and returns the red, green, and blue colors.
    * It also extracts the most significant byte and populates the reserved element of RgbQuad (unlike pelToRGB).
    *
    * @param pel      the picture element
    *
    * @return         RgbQuad object that contains the r,g,b colors and the most significant byte
    */
   public RgbQuad pelToRGBQ(int pel)
   {
      RgbQuad rgbq = new RgbQuad();

      rgbq.blue = pel & 0x00FF;
      rgbq.green = (pel >> 8) & 0x00FF;
      rgbq.red = (pel >> 16) & 0x00FF;
      rgbq.reserved = (pel >> 24) & 0x00FF;

      return rgbq;
   }

   /*
    * Takes red, green and blue color values plus an additional byte and returns a single 32-bit integer color.
    * pelToRGBQ(int pel) goes the other way.
    *
    * @param red        the red color value
    * @param green      the green color value
    * @param blue       the blue color value
    * @param reserved   an additional byte
    *
    * @return           one 32-bit integer color
    */
   public int rgbqToPel(int red, int green, int blue, int reserved)
   {
      return (reserved << 24) | (red << 16) | (green << 8) | blue;
   }

   /*
    * Takes in a color picture element (pel) and returns the grayscale pel using this formula:
    * Y = 0.3RED+0.589GREEN+0.111Blue
    *
    * @param pel        the color picture element
    *
    * @return           the grayscale pel
    */
   public int colorToGrayscale(int pel)
   {
      RgbQuad rgb = pelToRGB(pel);

      int lum = (int) Math.round(0.3 * (double) rgb.red + 0.589 * (double) rgb.green + 0.111 * (double) rgb.blue);

      return rgbToPel(lum, lum, lum);
   }

   /*
    * Oversees the running of the network.
    *
    * @param args             arguments from the command line
    */
   public static void main(String[] args)
   {
      String inFileName, outFileName;
      int i, j, k;
      int numberOfColors;
      int pel;
      int iByteVal, iColumn, iBytesPerRow, iPelsPerRow, iTrailingBits, iDeadBytes;

      // RBGQUAD
      int rgbQuad_rgbBlue;
      int rgbQuad_rgbGreen;
      int rgbQuad_rgbRed;
      int rgbQuad_rgbReserved;           //not used in this method

      // the color table
      int[] colorPallet = new int[256];  //reserve space for the largest possible color table

      DibDump dibdumper = new DibDump(); //needed to get to the byte swapping methods

      if (args.length > 0)
         inFileName = args[0];
      else
         inFileName = "test1.bmp";

      if (args.length > 1)
         outFileName = args[1];
      else
         outFileName = "test2.bmp";

      try
      {
         //opening the file that is the first command line parameter
         FileInputStream fstream = new FileInputStream(inFileName);

         //converting the input stream to a DataInputStream
         DataInputStream in = new DataInputStream(fstream);

         /* Reads in the BITMAPFILEHEADER
          *
          *     typedef struct tagBITMAPFILEHEADER {
          *         WORD    bfType;
          *         DWORD   bfSize;
          *         WORD    bfReserved1;
          *         WORD    bfReserved2;
          *         DWORD   bfOffBits;
          *    } BITMAPFILEHEADER, FAR *LPBITMAPFILEHEADER, *PBITMAPFILEHEADER;
          *
          *    bfType         Specifies the file type. It must be set to the signature word BM (0x4D42) to indicate bitmap.
          *    bfSize         Specifies the size, in bytes, of the bitmap file.
          *    bfReserved1    Reserved; set to zero
          *    bfReserved2    Reserved; set to zero
          *    bfOffBits      Specifies the offset, in bytes, from the BITMAPFILEHEADER structure to the bitmap bits
          */
         //reading in the values and converting to big endian
         bmpFileHeader_bfType = dibdumper.swapShort(in.readUnsignedShort());    // WORD
         bmpFileHeader_bfSize = dibdumper.swapInt(in.readInt());                // DWORD
         bmpFileHeader_bfReserved1 = dibdumper.swapShort(in.readUnsignedShort());    // WORD
         bmpFileHeader_bfReserved2 = dibdumper.swapShort(in.readUnsignedShort());    // WORD
         bmpFileHeader_bfOffBits = dibdumper.swapInt(in.readInt());                // DWORD

         //printing out the values
         System.out.printf("bfType=%2X bfSize=%d bfReserved1=%h bfReserved2=%h bfOffBits=%d\n",
                 bmpFileHeader_bfType,
                 bmpFileHeader_bfSize,
                 bmpFileHeader_bfReserved1,
                 bmpFileHeader_bfReserved2,
                 bmpFileHeader_bfOffBits);

         /* Reads in the BITMAPINFOHEADER
          *
          *       typedef struct tagBITMAPINFOHEADER{
          *             DWORD      biSize;
          *             LONG       biWidth;
          *             LONG       biHeight;
          *             WORD       biPlanes;
          *             WORD       biBitCount;
          *             DWORD      biCompression;
          *             DWORD      biSizeImage;
          *             LONG       biXPelsPerMeter;
          *             LONG       biYPelsPerMeter;
          *             DWORD      biClrUsed;
          *             DWORD      biClrImportant;
          *    } BITMAPINFOHEADER, FAR *LPBITMAPINFOHEADER, *PBITMAPINFOHEADER;
          *
          *
          *    biSize         Specifies the size of the structure, in bytes.
          *                   This size does not include the color table or the masks mentioned in the biClrUsed member.
          *                   See the Remarks section for more information.
          *    biWidth        Specifies the width of the bitmap, in pixels.
          *    biHeight       Specifies the height of the bitmap, in pixels.
          *                   If it's positive, the bitmap is a bottom-up DIB and its origin is the lower left corner.
          *                   If it's negative, the bitmap is a top-down DIB and its origin is the upper left corner. Also, biCompression must be
          *                      either BI_RGB or BI_BITFIELDS. Top-down DIBs cannot be compressed.
          *    biPlanes       Specifies the number of planes for the target device, and its value must be set to 1.
          *    biBitCount     Specifies the number of bits per pixel and determines the number of bits that define each pixel and
          *                   the maximum number of colors in the bitmap. It must also be one of the following values:
          *
          *    Value     Description
          *    1         The bitmap is monochrome, and bmiColors contains two entries. Each bit in the bitmap array represents a pixel, and
          *              the most significant bit is to the left in the image.
          *              If the bit is clear, the pixel is displayed with the color of the first entry in the bmiColors table.
          *              If the bit is set, the pixel has the color of the second entry in the table.
          *    2         The bitmap has four possible color values, and the most significant half-nibble is to the left in the image.
          *    4         The bitmap has a maximum of 16 colors, and bmiColors contains up to 16 entries.
          *              Each pixel in the bitmap is represented by a 4-bit index in the color table, and the most significant nibble is to the left in the image.
          *    8         The bitmap has a maximum of 256 colors, and the bmiColors member contains up to 256 entries. Each byte in the array represents a single pixel.
          *    16        The bitmap has a maximum of 2^16 colors.
          *              If biCompression is BI_RGB, bmiColors is NULL.
          *              Each WORD in the bitmap array represents a single pixel.
          *              The relative intensities of red, green, and blue are represented with 5 bits for each color component.
          *              The value for blue is in the least significant 5 bits, followed by 5 bits each for green and red.
          *              The most significant bit is not used.
          *              The bmiColors color table is used for optimizing colors used on palette-based devices and must contain the number of entries specified by biClrUsed.
          *    24        The bitmap has a maximum of 2^24 colors, and the bmiColors member is NULL.
          *              Each 3-byte triplet in the bitmap array represents the relative intensities of blue, green, and red, respectively, for a pixel.
          *              The bmiColors color table is used for optimizing colors used on palette-based devices, and must contain the number of entries specified by biClrUsed.
          *    32        The bitmap has a maximum of 2^32 colors. If biCompression is BI_RGB, bmiColors is NULL. Each DWORD in the bitmap array represents the relative intensities of blue, green, and red, respectively, for a pixel.
          *              The high byte in each DWORD is not used. The bmiColors color table is used for optimizing colors used on palette-based devices, and must contain the number of entries specified by biClrUsed.
          *              If biCompression is BI_BITFIELDS, the bmiColors color table contains three DWORD color masks that specify the red, green, and blue components, respectively, of each pixel.
          *              Each DWORD in the bitmap array represents a single pixel.
          *
          *    biCompression   Specifies the type of compression for a compressed bottom-up bitmap (top-down DIBs cannot be compressed).
          *                    This member can be one of the following values:
          *
          *                    Value               Description
          *                    BI_RGB              An uncompressed format
          *                    BI_BITFIELDS        Specifies that the bitmap is not compressed and that the color table consists of three DWORD color masks that specify the red, green, and blue components of each pixel.
          *                                        This is valid when used with 16- and 32-bpp bitmaps and in Windows Embedded CE versions 2.0 and later.
          *                    BI_ALPHABITFIELDS   Specifies that the bitmap is not compressed and that the color table consists of four DWORD color masks that specify the red, green, blue, and alpha components of each pixel.
          *                                        This is valid when used with 16- and 32-bpp bitmaps and in Windows CE .NET 4.0 and later.
          *
          *                    The values in the above table can be ORed with BI_SRCPREROTATE to specify that the source DIB section has the same rotation angle as the destination.
          *    biSizeImage     Specifies the size of the image in bytes (this value gives the number of bytes in each scan line which must be padded to insure the line is a multiple of 4 bytes
          *                    (it must align on a DWORD boundary) times the number of rows).
          *                    May be set to zero for BI_RGB bitmaps (so cannot be sure it will be set).
          *    biXPelsPerMeter Specifies the horizontal resolution, in pixels per meter, of the target device for the bitmap.
          *                    This value can be used to select a bitmap from a resource group that best matches the characteristics of the current device. **
          *    biYPelsPerMeter Specifies the vertical resolution, in pixels per meter, of the target device for the bitmap
          *    biClrUsed       Specifies the number of color indexes in the color table that are actually used by the bitmap.
          *                    If value is zero, the bitmap uses the maximum number of colors corresponding to the value of biBitCount for the compression mode specified by biCompression.
          *                    If biClrUsed is nonzero and biBitCount is less than 16, biClrUsed specifies the actual number of colors the graphics engine or device driver accesses.
          *                    If biBitCount is 16 or greater, biClrUsed specifies the size of the color table used to optimize performance of the system color palettes.
          *                    If biBitCount equals 16 or 32, the optimal color palette starts immediately following the three DWORD masks.
          *                    If the bitmap is a packed bitmap (a bitmap in which the bitmap array immediately follows the BITMAPINFO header and is referenced by a single pointer), biClrUsed must be either zero or the actual size of the color table.
          *    biClrImportant  Specifies the number of color indexes required for displaying the bitmap. If this value is zero, all colors are required.
          *
          *    The BITMAPINFO structure combines the BITMAPINFOHEADER structure and a color table to provide a complete definition of the dimensions and colors of a DIB.
          *    The information stored in biSize should be used to locate the color table in a BITMAPINFO structure, as follows: **
          *        pColor = ((LPSTR)pBitmapInfo + (WORD)(pBitmapInfo->bmiHeader.biSize));
          */

         //reading in and converting to big endian
         bmpInfoHeader_biSize = dibdumper.swapInt(in.readInt());              // DWORD
         bmpInfoHeader_biWidth = dibdumper.swapInt(in.readInt());              // LONG
         bmpInfoHeader_biHeight = dibdumper.swapInt(in.readInt());              // LONG
         bmpInfoHeader_biPlanes = dibdumper.swapShort(in.readUnsignedShort());  // WORD
         bmpInfoHeader_biBitCount = dibdumper.swapShort(in.readUnsignedShort());  // WORD
         bmpInfoHeader_biCompression = dibdumper.swapInt(in.readInt());              // DWORD
         bmpInfoHeader_biSizeImage = dibdumper.swapInt(in.readInt());              // DWORD
         bmpInfoHeader_biXPelsPerMeter = dibdumper.swapInt(in.readInt());              // LONG
         bmpInfoHeader_biYPelsPerMeter = dibdumper.swapInt(in.readInt());              // LONG
         bmpInfoHeader_biClrUsed = dibdumper.swapInt(in.readInt());              // DWORD
         bmpInfoHeader_biClrImportant = dibdumper.swapInt(in.readInt());              // DWORD

         //printing out the values
         System.out.printf("biSize=%d\nbiWidth=%d\nbiHeight=%d\nbiPlanes=%d\nbiBitCount=%d\nbiCompression=%d\nbiSizeImage=%d\nbiXPelsPerMeter=%d\nbiYPelsPerMeter=%d\nbiClrUsed=%d\nbiClrImportant=%d\n",
                 bmpInfoHeader_biSize,
                 bmpInfoHeader_biWidth,
                 bmpInfoHeader_biHeight,
                 bmpInfoHeader_biPlanes,
                 bmpInfoHeader_biBitCount,
                 bmpInfoHeader_biCompression,
                 bmpInfoHeader_biSizeImage,
                 bmpInfoHeader_biXPelsPerMeter,
                 bmpInfoHeader_biYPelsPerMeter,
                 bmpInfoHeader_biClrUsed,
                 bmpInfoHeader_biClrImportant);

         System.out.printf("\n");

         //height cannot be negative (since used to create arrays); if it's negative, it's made positive, and
         //topDownDIB flag is set to TRUE (indicating that the image stored on disc upsidedown)
         if (bmpInfoHeader_biHeight < 0)
         {
            topDownDIB = true;
            bmpInfoHeader_biHeight = -bmpInfoHeader_biHeight;
         }

         /*
          * Color table, doesn't exist for true color images
          *
          * typedef struct tagRGBQUAD {
          *       BYTE    rgbBlue;
          *       BYTE    rgbGreen;
          *       BYTE    rgbRed;
          *       BYTE    rgbReserved;
          * } RGBQUAD;
          *
          * typedef RGBQUAD FAR* LPRGBQUAD;
          */
         switch (bmpInfoHeader_biBitCount) // Determine the number of colors in the default color table
         {
            case 1:
               numberOfColors = 2;
               break;
            case 2:
               numberOfColors = 4;
               break;
            case 4:
               numberOfColors = 16;
               break;
            case 8:
               numberOfColors = 256;
               break;
            default:
               numberOfColors = 0; //no color table
         }

         System.out.printf("Color Depth = %d, %d\n", bmpInfoHeader_biBitCount, numberOfColors);
         /*
          * biClrUsed -  Specifies the number of color indexes in the color table that are actually used by the bitmap.
          *     If this value is zero, the bitmap uses the maximum number of colors corresponding to the value of the biBitCount member for the compression mode specified by biCompression.
          *     If biClrUsed is nonzero and the biBitCount member is less than 16, the biClrUsed member specifies the actual number of colors the graphics engine or device driver accesses.
          *     If biBitCount is 16 or greater, the biClrUsed member specifies the size of the color table used to optimize performance of the system color palettes.
          *     If biBitCount equals 16 or 32, the optimal color palette starts immediately following the three DWORD masks.
          *     If the bitmap is a packed bitmap (a bitmap in which the bitmap array immediately follows the BITMAPINFO header and is referenced by a single pointer), the biClrUsed member must be either zero or the actual size of the color table.
          */
         if (bmpInfoHeader_biClrUsed > 0)
            numberOfColors = bmpInfoHeader_biClrUsed;

         for (i = 0; i < numberOfColors; ++i) // Read in the color table (or not if numberOfColors is zero)
         {
            rgbQuad_rgbBlue = in.readUnsignedByte(); // lowest byte in the color
            rgbQuad_rgbGreen = in.readUnsignedByte();
            rgbQuad_rgbRed = in.readUnsignedByte(); // highest byte in the color
            rgbQuad_rgbReserved = in.readUnsignedByte();

            // Build the color from the RGB values. Since we declared the rgbQuad values to be int, we can shift and then OR the values
            // to build up the color. Since we are reading one byte at a time, there are no "endian" issues.

            colorPallet[i] = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
         }//for (i = 0; i < numberOfColors; ++i)

         /*
          * Rest of the bitmap, values interpreted based on the color depth
          *
          * numberOfColors = 2:   Each bit is a pel, so there are 8 pels per byte. The Color Table has only two values for "black" and "white"
          * numberOfColors = 4:   Each pair of bits is a pel, so there are 4 pels per byte. The Color Table has only four values
          * numberOfColors = 16;  Each nibble (4 bits) is a pel, so there are 2 pels per byte. The Color Table has 16 entries.
          * numberOfColors = 256; Each byte is a pel and the value maps into the 256 byte Color Table.
          *
          * Any other value is read in as "true" color.
          *
          * The BMP image is stored from bottom to top, meaning that the first scan line is the last scan line in the image.
          *
          * In the 32-bit format, each pixel in the image is represented by a series of four bytes of RGB stored as xBRG,
          * where the 'x' is an unused byte. For ALL image types each scan line is padded to an even 4-byte boundary.
          */

         imageArray = new int[bmpInfoHeader_biHeight][bmpInfoHeader_biWidth]; //array of pels

         /*
          * For each of the cases, outer loop loops over the rows (in reverse); the inner loop loops over the columns.
          */
         switch (bmpInfoHeader_biBitCount)
         {
            case 1: // each bit is a color, so there are 8 pels per byte.  Works
               /*
                * Each byte read in is 8 columns, so we need to break them out. We also have to deal with the case
                * where the image width is not an integer multiple of 8, in which case we will
                * have bits from part of the remaining byte. Each color is 1 bit which is masked with 0x01.
                * The screen ordering of the pels is High-Bit to Low-Bit, so the most significant element is first in the array of pels.
                */
               iBytesPerRow = bmpInfoHeader_biWidth / 8;
               iTrailingBits = bmpInfoHeader_biWidth % 8;

               iDeadBytes = iBytesPerRow;
               if (iTrailingBits > 0) ++iDeadBytes;
               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j)
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 8; ++k)     //get 8 pels from the one byte
                     {
                        iColumn = j * 8 + k;
                        pel = colorPallet[(iByteVal >> (7 - k)) & 0x01];
                        imageArray[i][iColumn] = pel;
                     }
                  }
                  if (iTrailingBits > 0) //pick up the trailing bits for images that are not mod 8 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < iTrailingBits; ++k)
                     {
                        iColumn = iBytesPerRow * 8 + k;
                        pel = colorPallet[(iByteVal >> (7 - k)) & 0x01];
                        imageArray[i][iColumn] = pel;
                     }
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               }
               break;

            case 2: // 4 colors, Each byte is 4 pels (2 bits each),  Should work, not tested.
               /*
                * Each byte read in is 4 columns, so we need to break them out. We also have to deal with the case
                * where the image width is not an integer multiple of 4, in which case we will
                * have from 2 to 6 bits of the remaining byte. Each color is 2 bits which is masked with 0x03.
                * The screen ordering of the pels is High-Half-Nibble to Low-Half-Nibble, so the most significant element is first in the array of pels.
                */
               iBytesPerRow = bmpInfoHeader_biWidth / 4;
               iTrailingBits = bmpInfoHeader_biWidth % 4; // 0, 1, 2 or 3

               iDeadBytes = iBytesPerRow;
               if (iTrailingBits > 0) ++iDeadBytes;
               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j)
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 4; ++k) // Get 4 pels from one byte
                     {
                        iColumn = j * 4 + k;
                        pel = colorPallet[(iByteVal >> ((3 - k) * 2)) & 0x03]; // shift 2 bits at a time and reverse order
                        imageArray[i][iColumn] = pel;
                     }
                  }
                  if (iTrailingBits > 0) // pick up the trailing nibble for images that are not mod 2 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < iTrailingBits; ++k)
                     {
                        iColumn = iBytesPerRow * 4 + k;
                        pel = colorPallet[(iByteVal >> ((3 - k) * 2)) & 0x03];
                        imageArray[i][iColumn] = pel;
                     }
                  }
                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               }
               break;

            case 4: // 16 colors, Each byte is two pels. Works
               /*
                * Each byte read in is 2 columns, so we need to break them out. We also have to deal with the case
                * where the image width is not an integer multiple of 2, in which case we will
                * have one nibble from part of the remaining byte. We then read in the dead bytes so that each
                * scan line is a multiple of 4 bytes. Each color is a nibble (4 bits) which is masked with 0x0F.
                * The screen ordering of the pels is High-Nibble Low-Nibble, so the most significant element is first in the array of pels.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iBytesPerRow = iPelsPerRow / 2;
               iTrailingBits = iPelsPerRow % 2;  // Will either be 0 or 1

               iDeadBytes = iBytesPerRow;
               if (iTrailingBits > 0) ++iDeadBytes;
               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j)
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 2; ++k) // Two pels per byte
                     {
                        iColumn = j * 2 + k;          // 1 - k  is needed to have High, Low nibble ordering for the image.
                        pel = colorPallet[(iByteVal >> ((1 - k) * 4)) & 0x0F]; // shift 4 bits at a time
                        imageArray[i][iColumn] = pel;
                     }
                  }

                  if (iTrailingBits > 0) // pick up the trailing nibble for images that are not mod 2 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     iColumn = iBytesPerRow * 2;
                     pel = colorPallet[(iByteVal >> 4) & 0x0F]; // The High nibble is the last remaining pel
                     imageArray[i][iColumn] = pel;
                  }
                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               } // for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)
               break;

            case 8: // 1 byte, 1 pel, Works
               /*
                * Each byte read in is 1 column. We then read in the dead bytes so that each scan line is a multiple of 4 bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - iPelsPerRow % 4) % 4;
               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j)         // j is now just the column counter
                  {
                     iByteVal = in.readUnsignedByte();
                     pel = colorPallet[iByteVal];
                     imageArray[i][j] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               }
               break;

            case 16: // Not likely to work (format is not internally consistent), not tested.
               /*
                * Each two bytes read in is 1 column. Each color is 5 bits in the 2 byte word value, so we shift 5 bits and then mask them
                * off with 0x1F which is %11111 in binary. We then read in the dead bytes so that each scan line is a multiple of 4 bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - iPelsPerRow % 4) % 4;
               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j)         // j is now just the column counter
                  {
                     pel = dibdumper.swapShort(in.readUnsignedShort()); // Need to deal with little endian values
                     rgbQuad_rgbBlue = pel & 0x1F;
                     rgbQuad_rgbGreen = (pel >> 5) & 0x1F;
                     rgbQuad_rgbRed = (pel >> 10) & 0x1F;
                     pel = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
                     imageArray[i][j] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               } // for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)
               break;

            case 24: // Works
               /*
                * Each three bytes read in is 1 column. Each scan line is padded to by a multiple of 4 bytes. The disk image has only 3 however.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - (iPelsPerRow * 3) % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j)         // j is now just the column counter
                  {
                     rgbQuad_rgbBlue = in.readUnsignedByte();
                     rgbQuad_rgbGreen = in.readUnsignedByte();
                     rgbQuad_rgbRed = in.readUnsignedByte();
                     pel = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
                     imageArray[i][j] = pel;
                  }
                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); // Now read in the "dead bytes" to pad to a 4 byte boundary
               }
               break;

            case 32: // Works
               /*
                * Each four bytes read in is 1 column. The number of bytes per line will always be a multiple of 4, so there are no dead bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) // read over the rows
               {
                  if (topDownDIB) i = row;
                  else i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j)         // j is now just the column counter
                  {
                     rgbQuad_rgbBlue = in.readUnsignedByte();
                     rgbQuad_rgbGreen = in.readUnsignedByte();
                     rgbQuad_rgbRed = in.readUnsignedByte();
                     rgbQuad_rgbReserved = in.readUnsignedByte();
                     pel = (rgbQuad_rgbReserved << 24) | (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
                     imageArray[i][j] = pel;
                  }
               }
               break;

            default:
               System.out.printf("This error should not occur - 1!\n");

         } // switch (bmpInfoHeader_biBitCount)

         in.close();
         fstream.close();
      } // try

      catch (Exception e)
      {
         System.err.println("File input error" + e);
      }

      /*
       * Console dump of image bytes in HEX if the image is smaller than 33 x 33
       */
      if ((bmpInfoHeader_biWidth < 33) && (bmpInfoHeader_biHeight < 33))
      {
         iBytesPerRow = bmpInfoHeader_biWidth;
         for (i = 0; i < bmpInfoHeader_biHeight; ++i) // read over the rows
         {
            for (j = 0; j < iBytesPerRow; ++j)         // j is now just the column counter
            {
               System.out.printf("%06X\t", imageArray[i][j]);
            }

            System.out.printf("\n");
         }
      }

      /*
       * Masking the pels (to get rid of the reserved value), scaling them from 0 to 1, and
       * writing them out to a file.
       */
      try
      {
         BufferedWriter pelsWriter = new BufferedWriter(new FileWriter("inputPels.txt"));

         for (int[] r : imageArray)
         {
            for (int p : r)
            {
               p = p & 0x00FFFFFF;
               pelsWriter.write((double)p/(Math.pow(2, 24)-1) + " ");
            }
         }

         pelsWriter.close();
      }

      catch (IOException e)
      {
         System.out.println("There was a problem writing the pels out to a file.");
      }

      System.out.println("\n"); //for formatting in the console

      /*
       * Training the neural network.
       */
      NeuralNet net = new NeuralNet(); //creates a neural net


      /*
       * Reading in the pel values from the file and scaling them back up to 0 to (2^24)-1.
       */
      int[][] outputtedPels = new int[bmpInfoHeader_biHeight][bmpInfoHeader_biWidth];

      String outputPelsFileName;

      if (args.length > 0)
         outputPelsFileName = args[0];
      else
         outputPelsFileName = "outputPels.txt";

      try
      {
         BufferedReader pelsReader = new BufferedReader(new FileReader(outputPelsFileName));

         StringTokenizer st = new StringTokenizer(pelsReader.readLine());

         for (int r = 0; r < outputtedPels.length; r++)
         {
            for (int c = 0; c < outputtedPels[0].length; c++)
            {
               outputtedPels[r][c] = (int)(Double.parseDouble(st.nextToken()) * (Math.pow(2, 24)-1));
            }
         }

         pelsReader.close();
      }

      catch (Exception e)
      {
         System.out.println("Exception occurred trying to read the pels");
         e.printStackTrace();
      }

      /*
       * Writing the pel values out to a bmp file.
       */
      try
      {
         iDeadBytes = (4 - (bmpInfoHeader_biWidth * 3) % 4) % 4;

         bmpInfoHeader_biSizeImage =  (bmpInfoHeader_biWidth * 3 + iDeadBytes) * bmpInfoHeader_biHeight;
         bmpFileHeader_bfOffBits = 54;        // 54 byte offset for 24 bit images (just open one with this app to get this value)
         bmpFileHeader_bfSize = bmpInfoHeader_biSizeImage + bmpFileHeader_bfOffBits;
         bmpInfoHeader_biBitCount = 24;       // 24 bit color image. Changes this number to 32 to write out 32 bit color images
         bmpInfoHeader_biCompression = 0;     // BI_RGB (which is a value of zero)
         bmpInfoHeader_biClrUsed = 0;         // Zero for true color
         bmpInfoHeader_biClrImportant = 0;    // Zero for true color

         FileOutputStream fstream = new FileOutputStream(outFileName);
         DataOutputStream out = new DataOutputStream(fstream);

// BITMAPFILEHEADER
         out.writeShort(dibdumper.swapShort(bmpFileHeader_bfType));      // WORD
         out.writeInt(dibdumper.swapInt(bmpFileHeader_bfSize));          // DWORD
         out.writeShort(dibdumper.swapShort(bmpFileHeader_bfReserved1)); // WORD
         out.writeShort(dibdumper.swapShort(bmpFileHeader_bfReserved2)); // WORD
         out.writeInt(dibdumper.swapInt(bmpFileHeader_bfOffBits));       // DWORD

// BITMAPINFOHEADER
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biSize));          // DWORD
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biWidth));         // LONG
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biHeight));        // LONG
         out.writeShort(dibdumper.swapShort(bmpInfoHeader_biPlanes));    // WORD
         out.writeShort(dibdumper.swapShort(bmpInfoHeader_biBitCount));  // WORD
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biCompression));   // DWORD
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biSizeImage));     // DWORD
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biXPelsPerMeter)); // LONG
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biYPelsPerMeter)); // LONG
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biClrUsed));       // DWORD
         out.writeInt(dibdumper.swapInt(bmpInfoHeader_biClrImportant));  // DWORD

// there is no color table for this true color image, so write out the pels

         rgbQuad_rgbReserved = 0;

         for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)    // write over the rows (in the usual inverted format)
         {
            for (j = 0; j < bmpInfoHeader_biWidth; ++j) // and the columns
            {
               pel = outputtedPels[i][j];
               rgbQuad_rgbBlue  = pel & 0x00FF;
               rgbQuad_rgbGreen = (pel >> 8)  & 0x00FF;
               rgbQuad_rgbRed   = (pel >> 16) & 0x00FF;
               out.writeByte(rgbQuad_rgbBlue); // lowest byte in the color
               out.writeByte(rgbQuad_rgbGreen);
               out.writeByte(rgbQuad_rgbRed);  // highest byte in the color
               if (bmpInfoHeader_biBitCount == 32) out.writeByte(rgbQuad_rgbReserved);
            }

            if (bmpInfoHeader_biBitCount != 32)
               {
               for (j = 0; j < iDeadBytes; ++j)
                  {
                  out.writeByte(0); // Now write out the "dead bytes" to pad to a 4 byte boundary
                  }
               }
            } // for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)

         out.close();
         fstream.close();
      }

      catch (Exception e)
      {
         System.err.println("File output error" + e);
      }
   }//public static void main(String[] args)
}//public class DipDump
