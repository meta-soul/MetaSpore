package com.dmetasoul.metaspore.feature.dao;


import java.util.List;

public class TableAttributes {
    private String dbType;
    private String tableName;
    private String collectionName;
    private List<Column> columns;

    public TableAttributes() {
    }

    public String getDbType() {
        return this.dbType;
    }

    public String getTableName() {
        return this.tableName;
    }

    public String getCollectionName() {
        return this.collectionName;
    }

    public List<Column> getColumns() {
        return this.columns;
    }

    public void setDbType(String dbType) {
        this.dbType = dbType;
    }

    public void setTableName(String tableName) {
        this.tableName = tableName;
    }

    public void setCollectionName(String collectionName) {
        this.collectionName = collectionName;
    }

    public void setColumns(List<Column> columns) {
        this.columns = columns;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof TableAttributes)) return false;
        final TableAttributes other = (TableAttributes) o;
        if (!other.canEqual((Object) this)) return false;
        final Object this$dbType = this.getDbType();
        final Object other$dbType = other.getDbType();
        if (this$dbType == null ? other$dbType != null : !this$dbType.equals(other$dbType)) return false;
        final Object this$tableName = this.getTableName();
        final Object other$tableName = other.getTableName();
        if (this$tableName == null ? other$tableName != null : !this$tableName.equals(other$tableName)) return false;
        final Object this$collectionName = this.getCollectionName();
        final Object other$collectionName = other.getCollectionName();
        if (this$collectionName == null ? other$collectionName != null : !this$collectionName.equals(other$collectionName))
            return false;
        final Object this$columns = this.getColumns();
        final Object other$columns = other.getColumns();
        if (this$columns == null ? other$columns != null : !this$columns.equals(other$columns)) return false;
        return true;
    }

    protected boolean canEqual(final Object other) {
        return other instanceof TableAttributes;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final Object $dbType = this.getDbType();
        result = result * PRIME + ($dbType == null ? 43 : $dbType.hashCode());
        final Object $tableName = this.getTableName();
        result = result * PRIME + ($tableName == null ? 43 : $tableName.hashCode());
        final Object $collectionName = this.getCollectionName();
        result = result * PRIME + ($collectionName == null ? 43 : $collectionName.hashCode());
        final Object $columns = this.getColumns();
        result = result * PRIME + ($columns == null ? 43 : $columns.hashCode());
        return result;
    }

    public String toString() {
        return "TableAttributes(dbType=" + this.getDbType() + ", tableName=" + this.getTableName() + ", collectionName=" + this.getCollectionName() + ", columns=" + this.getColumns() + ")";
    }
}
