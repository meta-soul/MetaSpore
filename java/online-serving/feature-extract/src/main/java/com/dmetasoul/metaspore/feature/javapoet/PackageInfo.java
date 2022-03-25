package com.dmetasoul.metaspore.feature.javapoet;

public class PackageInfo {
    public static final String PROJECT_ROOT_PATH = System.getProperty("user.dir") + "/target/generated-sources/feature/java";
    private String packageName;
    private String domain;
    private String repository;
    private String controller;

    public PackageInfo(String packageName) {
        this.packageName = packageName;
        this.domain = String.format("%s.domain", packageName);
        this.repository = String.format("%s.repository", packageName);
        this.controller = String.format("%s.controller", packageName);
    }

    public String getPackageName() {
        return this.packageName;
    }

    public String getDomain() {
        return this.domain;
    }

    public String getRepository() {
        return this.repository;
    }

    public String getController() {
        return this.controller;
    }

    public void setPackageName(String packageName) {
        this.packageName = packageName;
    }

    public void setDomain(String domain) {
        this.domain = domain;
    }

    public void setRepository(String repository) {
        this.repository = repository;
    }

    public void setController(String controller) {
        this.controller = controller;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof PackageInfo)) return false;
        final PackageInfo other = (PackageInfo) o;
        if (!other.canEqual((Object) this)) return false;
        final Object this$packageName = this.getPackageName();
        final Object other$packageName = other.getPackageName();
        if (this$packageName == null ? other$packageName != null : !this$packageName.equals(other$packageName))
            return false;
        final Object this$domain = this.getDomain();
        final Object other$domain = other.getDomain();
        if (this$domain == null ? other$domain != null : !this$domain.equals(other$domain)) return false;
        final Object this$repository = this.getRepository();
        final Object other$repository = other.getRepository();
        if (this$repository == null ? other$repository != null : !this$repository.equals(other$repository))
            return false;
        final Object this$controller = this.getController();
        final Object other$controller = other.getController();
        if (this$controller == null ? other$controller != null : !this$controller.equals(other$controller))
            return false;
        return true;
    }

    protected boolean canEqual(final Object other) {
        return other instanceof PackageInfo;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final Object $packageName = this.getPackageName();
        result = result * PRIME + ($packageName == null ? 43 : $packageName.hashCode());
        final Object $domain = this.getDomain();
        result = result * PRIME + ($domain == null ? 43 : $domain.hashCode());
        final Object $repository = this.getRepository();
        result = result * PRIME + ($repository == null ? 43 : $repository.hashCode());
        final Object $controller = this.getController();
        result = result * PRIME + ($controller == null ? 43 : $controller.hashCode());
        return result;
    }

    public String toString() {
        return "PackageInfo(packageName=" + this.getPackageName() + ", domain=" + this.getDomain() + ", repository=" + this.getRepository() + ", controller=" + this.getController() + ")";
    }
}
