package com.dmetasoul.metaspore.common;

import com.amazonaws.AmazonClientException;
import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.AmazonS3URI;
import com.amazonaws.services.s3.model.S3Object;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;

@Slf4j
public class S3Client {
    private static final String AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID";
    private static final String AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY";
    private static final String AWS_ENDPOINT = "AWS_ENDPOINT";
    private static final String AWS_REGION = "AWS_REGION";

    public static AWSCredentials getAWSCredentials() {
        String awsAccessKey = System.getenv(AWS_ACCESS_KEY_ID);
        String awsSecretAccessKey = System.getenv(AWS_SECRET_ACCESS_KEY);
        if (StringUtils.isEmpty(awsAccessKey) || StringUtils.isEmpty(awsSecretAccessKey)) {
            try {
                return new ProfileCredentialsProvider("default").getCredentials();
            } catch (Exception ignored) {
            }
        }
        log.info(String.format("credentials: AWS_ACCESS_KEY_ID：%s, AWS_SECRET_ACCESS_KEY:%s",
                awsAccessKey, awsSecretAccessKey));
        return new BasicAWSCredentials(awsAccessKey, awsSecretAccessKey);
    }

    public static AmazonS3 getAwsClient() {
        AWSCredentials credentials = getAWSCredentials();
        String endpoint = System.getenv(AWS_ENDPOINT);
        String region = System.getenv(AWS_REGION);
        AmazonS3ClientBuilder builder = AmazonS3ClientBuilder.standard()
                .withCredentials(new AWSStaticCredentialsProvider(credentials));
        if (StringUtils.isNotEmpty(endpoint)) {
            if (StringUtils.isEmpty(region)) {
                region = "cn-southwest-2";
            }
            log.info("endpoint={}, region={}", endpoint, region);
            builder.setEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(endpoint, region));
        }
        return builder.build();
    }

    public static InputStream downloadFromS3(AmazonS3 s3Client, AmazonS3URI amazonS3URI) throws IOException {
        return prepareInputStream(s3Client.getObject(amazonS3URI.getBucket(), amazonS3URI.getKey()).getObjectContent());
    }

    public static InputStream prepareInputStream(InputStream input) throws IOException {
        if (!input.markSupported()) {
            input = new BufferedInputStream(input);
        }

        if (isCompressed(input)) {
            input = gunzip(input);
        }

        return input;
    }

    public static InputStream gunzip(InputStream is) throws IOException {
        try {
            return new BufferedInputStream(new GZIPInputStream(is));
        } catch (ZipException z) {
            return is;
        }
    }

    public static boolean isCompressed(InputStream is) throws IOException {
        try {
            if (!is.markSupported()) {
                is = new BufferedInputStream(is);
            }
            byte[] bytes = new byte[2];
            is.mark(2);
            boolean empty = (is.read(bytes) == 0);
            is.reset();
            return empty | (bytes[0] == (byte) GZIPInputStream.GZIP_MAGIC && bytes[1] == (byte) (GZIPInputStream.GZIP_MAGIC >> 8));
        } catch (Exception e) {
            return false;
        }
    }

    @SneakyThrows
    public static String downloadModelByShell(String model, String version, String s3Path, String localPath) {
        Path target = Path.of(localPath, model, version);
        String cmd = String.format("aws s3 sync --delete %s %s", s3Path, target);
        Utils.runCmd(cmd, null);
        return target.toString();
    }

    @SneakyThrows
    public static String downloadModel(String model, String version, String s3Path, String localPath) {
        Path target = Path.of(localPath, model, version);
        URI pathToBeDownloaded = new URI(s3Path);
        AmazonS3URI s3URI = new AmazonS3URI(pathToBeDownloaded);
        AmazonS3 s3Client = getAwsClient();
        // to do aws s3 download
        downloadFile(s3Client, s3URI, target.toString());
        return target.toString();
    }

    private static void downloadFile(AmazonS3 s3Client, AmazonS3URI s3URI, String fileName) {
        try {
            InputStream inputStream = downloadFromS3(s3Client, s3URI);
            FileUtils.saveToFile(inputStream, fileName);
        } catch (Exception e) {
            log.error("s3 file load fail! s3path: bucket:{}, key:{}, local: {}", s3URI.getBucket(), s3URI.getKey(), fileName);
            e.printStackTrace();
        }
    }
}
