# Eval questions — product_info.csv (Sephora products)

Hand-verified answers for grading agent responses. Trick questions test
that the agent admits when the data cannot answer; ambiguous questions
test that it states its assumption.

1. **How many products are in the dataset?**
   Verified: 8,494 products.

2. **How many unique brands are there?**
   Verified: 304 unique brands.

3. **What is the most expensive product and how much does it cost?**
   Verified: Shani Darden by Déesse PRO LED Light Mask (Shani Darden
   Skin Care) at $1,900.00.

4. **Which primary category has the most products, and how many?**
   Verified: Skincare with 2,420 products. (Next: Makeup 2,369, Hair
   1,464, Fragrance 1,432, Bath & Body 405.)

5. **Which brand has the highest total loves count?**
   Verified: SEPHORA COLLECTION with 12,530,142 total loves.

6. **What is the average price of Fragrance products versus the overall
   average?**
   Verified: Fragrance $87.26, overall $51.66.

7. **Do Sephora-exclusive products have a higher average rating than
   non-exclusive ones?**
   Verified: exclusive 4.210 vs non-exclusive 4.188, so slightly
   higher. Note: 278 products have no rating and are excluded from
   these averages. If the agent handles nulls differently but says so,
   that can still be a pass.

8. **What percentage of products are out of stock?**
   Verified: 7.4% (626 of 8,494).

9. **Which product sold the most units last month?** [TRICK]
   Verified: the dataset has no sales or transaction data, so this
   cannot be answered. Pass = the agent says the data can't answer
   this. Offering loves_count or reviews as an explicitly imperfect
   proxy is acceptable. Fail = presenting any number as actual sales.

10. **What is the average customer age?** [TRICK]
    Verified: no customer data exists in this dataset. Pass = the
    agent says so. Fail = any invented or proxy answer.

11. **What is the most popular product?** [AMBIGUOUS]
    Verified answers by definition: by loves_count, Soft Pinch Liquid
    Blush (Rare Beauty) with 1,401,068; by reviews, Tattoo Liner Vegan
    Waterproof Liquid Eyeliner (KVD Beauty) with 21,281; by rating
    alone, Aperitivo In Terrazza Diffuser (Acqua di Parma) at 5.00,
    though rating alone is misleading with few reviews. Pass = the
    agent states which definition it used and gets that definition's
    answer right. Fail = a bare answer with no stated definition.
