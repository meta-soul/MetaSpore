package com.dmetasoul.metaspore.common;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * 文件处理工具类
 * @author qinyy
 * @since 1.0, 2022-11-04
 */
@Slf4j
public class FileUtils {

    public static void saveToFile(InputStream is, String fileName) throws IOException{
        BufferedInputStream in;
        BufferedOutputStream out;
        in=new BufferedInputStream(is);
        out=new BufferedOutputStream(new FileOutputStream(fileName));
        int len=-1;
        byte[] b=new byte[10240];
        while((len=in.read(b))!=-1){
            out.write(b,0,len);
        }
        in.close();
        out.close();
    }

    public static boolean writeToFile(String fileName, String content, boolean append) {
        File file = new File(fileName);
        if (!canWrite(file)) {
            log.error("fileName: {} can not write", fileName);
            return false;
        }
        try (FileWriter fw = new FileWriter(file, append)){
            fw.write(content);
            fw.flush();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    public static void printTree(File path, String prefix) {
        log.error("dir {}{}", prefix, path.getName());
        try {
            File[] listFiles = path.listFiles();
            if (listFiles != null) {
                for (File item : listFiles) {
                    printTree(item, String.format("++%s", prefix));
                }
            }
        } catch (Exception e) {
            log.error("list path :{} fail", path);
        }
    }

    public static Boolean canRead(String path) {
        File file = new File(path);
        if (file.isDirectory()) {
            //printTree(file, "");
            try {
                File[] listFiles = file.listFiles();
                return listFiles != null;
            } catch (Exception e) {
                return false;
            }
        } else if (!file.exists()) {
            return false;
        }
        return checkRead(file);
    }

    public static String readFile(String path, Charset encoding) throws IOException
    {
        if (!canRead(path)) {
            return null;
        }
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }

    private static boolean checkRead(File file) {
        try (FileReader fd = new FileReader(file)) {
            if ((fd.read()) != -1) {
                return true;
            }
        } catch (IOException e) {
            return false;
        }
        return false;
    }

    public static Boolean canWrite(File file) {
        if (file.isDirectory()) {
            try {
                file = new File(file, "testWriteDeleteOnExit.temp");
                if (file.exists()) {
                    boolean checkWrite = checkWrite(file);
                    if (!deleteFile(file)) {
                        file.deleteOnExit();
                    }
                    return checkWrite;
                } else if (file.createNewFile()) {
                    if (!deleteFile(file)) {
                        file.deleteOnExit();
                    }
                    return true;
                } else {
                    return false;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
                // return false;
            }
        }
        return checkWrite(file);
    }

    private static boolean checkWrite(File file) {
        try (FileWriter fw = new FileWriter(file, true)){
            fw.write("");
            fw.flush();
            return true;
        } catch (IOException e) {
            throw new RuntimeException(e);
            // return false;
        } finally {
            if (!file.exists()) {
                deleteFile(file);
            }
        }
    }

    public static boolean deleteFile(File file) {
        return deleteFile(file, true);
    }

    public static boolean deleteFile(File file, boolean delDir) {
        if (!file.exists()) {
            return true;
        }
        if (file.isFile()) {
            return file.delete();
        } else {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    if (!deleteFile(child, delDir)) {
                        return false;
                    }
                }
            }
            if (delDir) {
                return file.delete();
            }
            return true;
        }
    }
}
