//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.dmetasoul.metaspore.feature.dao;

public class Column {
    private String colName;
    private String colType;

    public Column() {
    }

    public String getColName() {
        return this.colName;
    }

    public String getColType() {
        return this.colType;
    }

    public void setColName(String colName) {
        this.colName = colName;
    }

    public void setColType(String colType) {
        this.colType = colType;
    }

    public boolean equals(final Object o) {
        if (o == this) return true;
        if (!(o instanceof Column)) return false;
        final Column other = (Column) o;
        if (!other.canEqual((Object) this)) return false;
        final Object this$colName = this.getColName();
        final Object other$colName = other.getColName();
        if (this$colName == null ? other$colName != null : !this$colName.equals(other$colName)) return false;
        final Object this$colType = this.getColType();
        final Object other$colType = other.getColType();
        if (this$colType == null ? other$colType != null : !this$colType.equals(other$colType)) return false;
        return true;
    }

    protected boolean canEqual(final Object other) {
        return other instanceof Column;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        final Object $colName = this.getColName();
        result = result * PRIME + ($colName == null ? 43 : $colName.hashCode());
        final Object $colType = this.getColType();
        result = result * PRIME + ($colType == null ? 43 : $colType.hashCode());
        return result;
    }

    public String toString() {
        return "Column(colName=" + this.getColName() + ", colType=" + this.getColType() + ")";
    }
}