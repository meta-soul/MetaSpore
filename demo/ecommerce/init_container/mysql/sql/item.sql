CREATE TABLE IF NOT EXISTS `item`(
    `item_id` VARCHAR(100),
    `brand` text,
    `category` text,
    `category_levels` text,
    `description` text,
    `price` text,
    `title` text,
    `url` text,
    `image` text,
PRIMARY KEY ( `item_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;