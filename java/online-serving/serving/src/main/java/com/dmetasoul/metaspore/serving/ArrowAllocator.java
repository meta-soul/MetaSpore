package com.dmetasoul.metaspore.serving;

import org.apache.arrow.memory.RootAllocator;

public class ArrowAllocator {
    private static final RootAllocator allocator = new RootAllocator();

    public static RootAllocator getAllocator() {
        return ArrowAllocator.allocator;
    }
}
