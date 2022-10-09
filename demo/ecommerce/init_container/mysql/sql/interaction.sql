CREATE TABLE IF NOT EXISTS `interaction`(
    `item_id` VARCHAR(100),
    `rating` DOUBLE,
    `timestamp` INTEGER,
    `user_id` VARCHAR(100)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;